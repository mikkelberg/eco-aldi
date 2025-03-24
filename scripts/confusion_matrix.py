import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from sklearn.metrics import confusion_matrix, precision_recall_curve, confusion_matrix, f1_score, classification_report, average_precision_score


from utils import detectron_model as model
from utils import coco as cc

def iou(box1, box2):
    x1, y1, x2, y2 = box1  # Unpack coordinates for box1
    x1g, y1g, x2g, y2g = box2  # Unpack coordinates for box2

    # Compute the coordinates of the intersection rectangle
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)

    # Compute intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Compute union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    return inter_area / union_area if union_area > 0 else 0.0

def match_detections(ground_truth_dict: dict, pred_dict: dict, coco_json: dict, images_dir: str):
    print("Generating true and predicted bounding boxes and classes...")
    ground_truths = []
    predictions = []
    scores = []
    no_of_images = len(coco_json["images"])

    for idx, img in enumerate(coco_json["images"], start=1):
        if idx % 100 == 0 or idx == no_of_images:
            print(f"-- Processed {idx} of {no_of_images} in the test set.")
        pred_classes = pred_dict[img["id"]]["pred_classes"] 
        pred_scores = pred_dict[img["id"]]["pred_scores"]
        pred_bboxes = pred_dict[img["id"]]["pred_boxes"]
        true_labels = ground_truth_dict[img["id"]]

        matched = set()

        # No objects in this image
        if not true_labels: 
            if not pred_classes:
                # nothing was detected --> true negative
                ground_truths.append(-1)
                predictions.append(-1)
                scores.append(0.0)
                continue
            else:
                # something WAS detected --> false positive
                for cls, score in zip(pred_classes, pred_scores):
                    ground_truths.append(-1)
                    predictions.append(cls)
                    scores.append(score)
                continue

        # There ARE objects in this image
        for true_bbox, true_cls in true_labels:
            true_bbox = cc.coco_bbox_to_xyxy(bbox=true_bbox)
            best_iou = 0
            best_pred_idx = -1

            # Match predicted bboxes with the true bbox based on the best iou (largest overlap)
            for i, pred_bbox in enumerate(pred_bboxes):
                this_iou = iou(pred_bbox, true_bbox)
                if this_iou > best_iou:
                    best_iou = this_iou
                    best_pred_idx = i
            '''
            for i, score in enumerate(pred_scores):
                if score > best_score and i not in matched: # note that the predictions are all with a score above 0.5 (the threshold), so we don't have a check for that here
                    best_score = score
                    best_pred_idx = i'''
           
            if best_pred_idx == -1: 
                # no predicted bbox was found for this ground truth bbox --> false negative
                ground_truths.append(true_cls-1)
                predictions.append(-1)
                scores.append(0.0)
            else: 
                # we found a predicted bbox for this true bbox --> true positive
                ground_truths.append(true_cls-1)
                predictions.append(pred_classes[best_pred_idx]) # (but the class might still be wrong)
                scores.append(pred_scores[best_pred_idx])
                matched.add(best_pred_idx)                      # remember that we already assigned this prediction to a true label

        # Predicted bboxes that have not been matched to a true bbox --> false positive
        for j in range(len(pred_scores)):
            if j not in matched:
                ground_truths.append(-1)
                predictions.append(pred_classes[j])
                scores.append(pred_scores[j])

    print("Finished generating true and predicted bounding boxes and classes.")
    return ground_truths, predictions, scores

def plot_confusion_matrix(y_true, y_pred, coco_categories, output_dir):
    print("Plotting confusion matrix...")
    num_classes = len(coco_categories)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)) + [-1])
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise

    labels = [cat["name"] for cat in coco_categories] + ["no object"] # Convert to readable class names
    
    # plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir+"confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix to {output_dir}confusion_matrix.png")

def plot_precision_recall(y_true, y_pred, y_scores, coco_categories, output_dir):
    class_labels = [cat["name"] for cat in coco_categories] + ["no object"]
    class_ids = list(range(len(coco_categories))) + [-1]
    # For each class, compute precision-recall curve and plot
    plt.figure(figsize=(8, 6))
    for class_id in range(len(class_ids)):
        # Binarize the labels for each class
        y_true_class = [1 if t == class_id else 0 for t in y_true]
        y_scores_class = [score if pred == class_id else 0.0 for pred, score in zip(y_pred, y_scores)]
        
        # Only compute the precision-recall curve if there are positive samples
        if any(y_true_class):  # Only proceed if there are positive samples for this class
            precision, recall, _ = precision_recall_curve(y_true_class, y_scores_class)
            plt.plot(recall, precision, marker='.', label=class_labels[class_id])
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Per Class)")
    plt.legend()
    plt.savefig(output_dir+"precision_recall_curve.png", bbox_inches='tight')
    plt.close()

def plot_and_save_results(coco_json_path, image_dir, predictions_path, output_dir):
    coco_json = cc.load_from_file(coco_json_path)
    pred_dict = cc.load_from_file(predictions_path)
    ground_truth_dict = cc.load_image_to_bbox_and_cat_pairs_from_annotations(coco_json=coco_json)
    ground_truth_classes, predicted_classes, scores = match_detections(ground_truth_dict=ground_truth_dict, pred_dict=pred_dict, images_dir=image_dir, coco_json=coco_json)
    
    plot_confusion_matrix(y_true=ground_truth_classes, y_pred=predicted_classes, coco_categories=coco_json["categories"], output_dir=output_dir)
    plot_precision_recall(y_true=ground_truth_classes, y_pred=predicted_classes, y_scores=scores, coco_categories=coco_json["categories"], output_dir=output_dir)

def main():
     # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    IMAGE_FOLDER = "/mnt/data0/martez/" + args.dataset_name + "/images/"   # Folder containing test images
    COCO_JSON = "/mnt/data0/martez/" + args.dataset_name + "/annotations/" + args.dataset_name + "_test.json"
    OUTPUT_DIR = "results/"
    PREDS_FILE = "results/predictions.json"

    plot_and_save_results(coco_json_path=COCO_JSON, image_dir=IMAGE_FOLDER, predictions_path=PREDS_FILE, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()