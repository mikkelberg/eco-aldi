from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

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