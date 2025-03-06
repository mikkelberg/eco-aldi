import json
import os 
from PIL import Image

def save_json_to_file(json_obj, path):
    with open(path, "w") as f:
        json.dump(json_obj, f, indent=4)

def load_json_from_file(path):
    with open(path, "r") as f:
        json_obj = json.load(f)
    return json_obj

def read_image_size(image_path):
    if os.path.exists(image_path):
        with Image.open(image_path) as image:
            return image.size # width, height
    else:
        raise FileNotFoundError(f"Warning: Image not found - {image_path}")