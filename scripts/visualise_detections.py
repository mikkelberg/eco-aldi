import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
import argparse

def load_model(config_path, model_weights, score_thresh=0.5):
    cfg = get_cfg()
    try:
        cfg.merge_from_file(config_path)  # Load model configuration
    except KeyError as e:
        print(f"Warning: Skipping unknown config key: {e}")  # Ignore unknown keys
    cfg.MODEL.WEIGHTS = model_weights  # Load trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh  # Set confidence threshold for detection
    return DefaultPredictor(cfg)# , MetadataCatalog.get(cfg.DATASETS.TEST[0])

def load_metadata(dataset, coco_json):
    metadata = MetadataCatalog.get(dataset)  # Get dataset metadata
    if not hasattr(metadata, "thing_classes") or not metadata.thing_classes:
        with open(coco_json, "r") as f:
            coco_data = json.load(f)
        metadata.thing_classes = [cat["name"] for cat in coco_data["categories"]]
    return metadata

# Load the COCO JSON file
def load_coco_json(coco_json_path):
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    return coco_data

# Load image by its ID
def load_image(image_id, images_info, image_dir):
    img_info = next(item for item in images_info if item['id'] == image_id)
    img_path = f"{image_dir}/{img_info['file_name']}"
    image = cv2.imread(img_path)
    return image, img_info

def draw_bboxes(image, bboxes, labels, category_names, color=(0, 255, 0), thickness=5):
    for bbox, label in zip(bboxes, labels):
        x, y, w, h = list(map(int, bbox))
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)
        cv2.putText(image, f"Class: {category_names[label]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Get random samples for each class and negative class
def get_random_samples(coco_data, num_samples=1):
    annotations_by_class = defaultdict(list)
    for annotation in coco_data['annotations']:
        annotations_by_class[annotation['category_id']].append(annotation)
    
    # Pick random samples for each class
    selected_samples = []
    all_category_ids = list(annotations_by_class.keys())
    
    # Select 3 random samples for each class
    for cat_id in all_category_ids:
        samples = random.sample(annotations_by_class[cat_id], num_samples)
        selected_samples.extend(samples)
    
    # Negative samples: Select images without any annotation
    image_ids_with_annotations = set(anno['image_id'] for anno in coco_data['annotations'])
    all_image_ids = [image['id'] for image in coco_data['images']]
    negative_image_ids = list(set(all_image_ids) - image_ids_with_annotations)
    
    negative_samples = []
    for _ in range(num_samples):
        img_id = random.choice(negative_image_ids)
        negative_samples.append({'image_id': img_id, 'category_id': 0, 'bbox': []})
    
    selected_samples.extend(negative_samples)
    
    return selected_samples

def predict_bboxes(image, predictor):
    # Perform inference on the image
    outputs = predictor(image)
    
    # Extract the bounding boxes, labels, and scores
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()
    labels = outputs['instances'].pred_classes.cpu().numpy()

    # Convert the bounding boxes from tensor format [x_min, y_min, x_max, y_max]
    predicted_bboxes = boxes.tolist()
    
    # Filter predictions based on confidence scores (for example, confidence > 0.5)
    threshold = 0.5
    filtered_bboxes = []
    filtered_labels = []
    for i, score in enumerate(scores):
        if score > threshold:
            filtered_bboxes.append(predicted_bboxes[i])
            filtered_labels.append(labels[i])
    
    return filtered_bboxes, filtered_labels

# Main function to create the matrix of images
def create_image_matrix(coco_json_path, image_dir, output_path, predictor, num_samples=1):
    coco_data = load_coco_json(coco_json_path)
    catid_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    selected_samples = get_random_samples(coco_data, num_samples)

    fig, axes = plt.subplots(len(selected_samples), 3, figsize=(3*45, len(selected_samples)*20))
    
    for idx, sample in enumerate(selected_samples):
        # Load the image corresponding to the sample
        img_id = sample['image_id']
        image, img_info = load_image(img_id, coco_data['images'], image_dir)
        
        # Get ground truth bboxes
        gt_bboxes = []
        gt_labels = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == img_id:
                gt_bboxes.append(annotation['bbox'])
                gt_labels.append(annotation['category_id'])
        
        # Get predicted bboxes and labels using Detectron2
        pred_bboxes, pred_labels = predict_bboxes(image, predictor)
        
        # Draw GT bboxes (Green)
        gt_image = image.copy()
        gt_image = draw_bboxes(image=gt_image, bboxes=gt_bboxes, labels=gt_labels, color=(255, 0, 0), category_names=catid_to_name)  # Blue for GT
        
        # Draw predicted bboxes (Red)
        pred_image = image.copy()
        pred_image = draw_bboxes(image=pred_image, bboxes=pred_bboxes, labels=pred_labels, color=(0, 0, 255),category_names=catid_to_name)  # Red for predictions
        
        # Show images in the matrix
        fig, axes = plt.subplots(1, 3, figsize=(3*45, 20))

        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Prediction")
        axes[1].axis('off')

        # Show original image for reference
        axes[2].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Ground Truth")
        axes[2].axis('off')
    
        # Save the final matrix as a PNG file
        plt.tight_layout()
        plt.savefig(output_path+"/example_detections_" + str(idx) + ".png")
        plt.close()

def main():
     # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    parser.add_argument("config_file", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    MODEL_WEIGHTS = "output/" + args.dataset_name + "_val_model_best.pth"  
    IMAGE_FOLDER = "/mnt/data0/martez/" + args.dataset_name + "/images/"   # Folder containing test images
    COCO_JSON = "/mnt/data0/martez/" + args.dataset_name + "/annotations/" + args.dataset_name + "_test.json"
    CONFIG_PATH = args.config_file
    OUTPUT_DIR = "eval-results/"

    predictor = load_model(CONFIG_PATH, MODEL_WEIGHTS)
    create_image_matrix(coco_json_path=COCO_JSON, image_dir=IMAGE_FOLDER, predictor=predictor, output_path=OUTPUT_DIR)


if __name__ == "__main__":
    main()
