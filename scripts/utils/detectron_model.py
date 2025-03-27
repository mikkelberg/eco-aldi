from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import json
import cv2
import os

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

def predict_bbox_and_class_with_score(predictor: DefaultPredictor, image_path: str):
    image = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Could not find image at path, {image_path}")

    outputs = predictor(image)
    pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    pred_classes = outputs['instances'].pred_classes.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()

    return pred_boxes, pred_classes, scores

def save_predictions_to_file(image_id, pred_boxes, pred_classes, scores, dest):
    if not os.path.isfile(dest):
        predictions = {}
    else:    
        with open(dest, "r") as f:
            predictions = json.load(f)
    
    predictions[image_id] = {
        "pred_boxes": pred_boxes.tolist(),
        "pred_classes": pred_classes.tolist(),
        "pred_scores": scores.tolist()
        }

    with open(dest, "w") as ff:
        json.dump(predictions, ff, indent=4)