from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import json
import cv2

def load_model(config_path, model_weights, score_thresh=0.5):
    
    # Load model configuration
    cfg = get_cfg()
    try:
        cfg.merge_from_file(config_path) 
    except KeyError as e:
        print(f"Warning: Skipping unknown config key: {e}")     
        # Ignore unknown keys (there are some that are custom made 
        # for ALDI-training, but not used/needed here :)
    
    cfg.MODEL.WEIGHTS = model_weights                           # Load trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh        # Set confidence threshold for detection
    return DefaultPredictor(cfg)

# Compute IoU between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def predict_bbox_and_class_with_score(predictor: DefaultPredictor, image_path: str):
    image = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Could not find image at path, {image_path}")

    outputs = predictor(image)
    pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    pred_classes = outputs['instances'].pred_classes.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()

    return pred_boxes, pred_classes, scores

def save_predictions_to_file(image_id, pred_boxes, pred_classes, scores, dest):
    with open(dest, "r") as f:
        predictions = json.load(f)
    
    predictions["predictions"].append(
        {image_id: {
        "pred_boxes": pred_boxes.tolist(),
        "pred_classes": pred_classes.tolist(),
        "pred_scores": scores.tolist()}})

    with open(dest, "w") as ff:
        json.dump(predictions, ff, indent=4)