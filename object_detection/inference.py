import argparse

import cv2
import numpy as np
from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

def draw_instance_predictions(vsl, predictions, score_threshold=None):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, vsl.metadata.get("thing_classes", None))
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    
    masks = predictions.pred_masks.tolist() if predictions.has("pred_masks") else None

    if score_threshold != None:
        top_id = np.where(scores.numpy()>score_threshold)[0].tolist()
        scores = torch.tensor(scores.numpy()[top_id])
        boxes.tensor = torch.tensor(boxes.tensor.numpy()[top_id])
        classes = [classes[ii] for ii in top_id]
        labels = [labels[ii] for ii in top_id]
        masks = [masks[ii] for ii in top_id]

    if masks:
        masks = np.asarray(masks)
        masks = [GenericMask(x, vsl.output.height, vsl.output.width) for x in masks]

    if vsl._instance_mode == ColorMode.SEGMENTATION and vsl.metadata.get("thing_colors"):
        colors = [
            vsl._jitter([x / 255 for x in vsl.metadata.thing_colors[c]]) for c in classes
        ]
        alpha = 0.8
    else:
        colors = None
        alpha = 0.5

    if vsl._instance_mode == ColorMode.IMAGE_BW:
        vsl.output.reset_image(
            vsl._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
        )
        alpha = 0.3

    vsl.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return vsl.output

def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file_name",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--score_threshold",
        help="Threshold of score to display",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    print(args.opts)
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    img = cv2.imread(args.image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text","title","list","table","figure"])

    output = predictor(img)["instances"]
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = draw_instance_predictions(v, output.to("cpu"), args.score_threshold)
    result_image = result.get_image()[:, :, ::-1]

    # step 6: save
    cv2.imwrite(args.output_file_name, result_image)

if __name__ == '__main__':
    main()

