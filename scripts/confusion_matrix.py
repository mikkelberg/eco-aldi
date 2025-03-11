import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from sklearn.metrics import confusion_matrix

from utils import detectron_model as model
from utils import coco as cc

def get_true_and_pred_labels(ground_truth_dict: dict, pred_dict: dict, coco_json: dict, images_dir: str):
    print("Generating true and predicted bounding boxes and classes...")
    ground_truths = []
    predictions = []
    no_of_images = len(coco_json["images"])

    for idx, img in enumerate(coco_json["images"], start=1):
        if idx % 100 == 0 or idx == no_of_images:
            print(f"-- Processed {idx} of {no_of_images} in the test set.")
        pred_classes = pred_dict[img["id"]]["pred_classes"] 
        pred_scores = pred_dict[img["id"]]["pred_scores"]
        true_labels = ground_truth_dict[img["id"]]

        matched = set()

        # No objects in this image
        if not true_labels: 
            if not pred_classes:
                # nothing was detected --> true negative
                ground_truths.append(-1)
                predictions.append(-1)
                continue
            else:
                # something WAS detected --> false positive
                for cls in pred_classes:
                    ground_truths.append(-1)
                    predictions.append(cls)
                continue

    
        # There ARE objected in this image
        for true_bbox, true_cls in true_labels:
            true_bbox = cc.coco_bbox_to_xyxy(bbox=true_bbox)
            best_score = 0
            best_pred_idx = -1

            # Match predicted bboxes with the true bbox based on the best score (largest overlap)
            for i, score in enumerate(pred_scores):
                if score > best_score and i not in matched: # note that the predictions are all with a score above 0.5 (the threshold), so we don't have a chek for that here
                    best_score = score
                    best_pred_idx = i
           
            if best_pred_idx == -1: 
                # no pred was found for this ground truth bbox --> false negative
                ground_truths.append(true_cls)
                predictions.append(-1)
            else: 
                # we found a predicted bbox for this true bbox --> true positive
                ground_truths.append(true_cls)
                predictions.append(pred_classes[best_pred_idx]) # (but the class might still be wrong)
                matched.add(best_pred_idx)                      # remember that we already assigned this prediction to a true label

        # Predicted bboxes that have not been matched to a true bbox --> false positive
        for j in range(len(pred_scores)):
            if j not in matched:
                ground_truths.append(-1)
                predictions.append(pred_classes[j])

    print("Finished generating true and predicted bounding boxes and classes.")
    return ground_truths, predictions

def plot_confusion_matrix(y_true, y_pred, coco_categories, output_path):
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
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def plot_and_save_confusion_matrix(coco_json_path, image_dir, predictions_path, output_path):
    pred_dict = cc.load_from_file(predictions_path)
    coco_json = cc.load_from_file(coco_json_path)
    ground_truth_dict = cc.load_image_to_bbox_and_cat_pairs_from_annotations(coco_json=coco_json)
    ground_truth_classes, predicted_classes = get_true_and_pred_labels(ground_truth_dict=ground_truth_dict, pred_dict=pred_dict, images_dir=image_dir, coco_json=coco_json)
    plot_confusion_matrix(y_true=ground_truth_classes, y_pred=predicted_classes, coco_categories=coco_json["categories"], output_path=output_path)


def main():
     # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    IMAGE_FOLDER = "/mnt/data0/martez/" + args.dataset_name + "/images/"   # Folder containing test images
    COCO_JSON = "/mnt/data0/martez/" + args.dataset_name + "/annotations/" + args.dataset_name + "_test.json"
    OUTPUT_PATH = "results/confusion_matrix.png"
    PREDS_FILE = "results/predictions.json"

    plot_and_save_confusion_matrix(coco_json_path=COCO_JSON, image_dir=IMAGE_FOLDER, predictions_path=PREDS_FILE, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()