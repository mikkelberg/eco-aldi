import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from sklearn.metrics import confusion_matrix, precision_recall_curve, confusion_matrix, f1_score, classification_report, average_precision_score, auc


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

def match_detections(ground_truth_dict: dict, pred_dict: dict, iou_thresh = 0.5, conf_thresh = 0.5):
    print("Generating true and predicted bounding boxes and classes...")
    ground_truths = []
    predictions = []
    conf_scores = []


    for img_id, preds in pred_dict.items():
        matched = set()
        for pred_bbox, pred_class, pred_score in zip(preds["pred_boxes"], preds["pred_classes"], preds["pred_scores"]):
            if pred_score < conf_thresh: 
                # skip predictions below the confidence threshold
                continue
            
            if img_id not in ground_truth_dict: 
                # no objects in this image --> false negative (handled after loop)
                continue
            # else, the image DOES have objects, so we consider each prediction for this image and compare 
            # it to the ground truth:
            best_iou, best_gt_idx = 0, -1
            matched = set()
            # go through the ground truth bboxes and see if they match this prediction (have sufficiently high IoU)
            for i, (true_bbox, true_cls) in enumerate(ground_truth_dict[img_id]):
                if i in matched: continue # skip the already matched gt boxes
                true_bbox = cc.coco_bbox_to_xyxy(bbox=true_bbox)
                this_iou = iou(pred_bbox, true_bbox)
                if this_iou > best_iou and pred_score > conf_thresh:
                    best_iou = this_iou
                    best_gt_idx = i 
            
            if best_iou >= iou_thresh:
                # we found a predicted bbox for this true bbox --> true positive
                ground_truths.append(ground_truth_dict[img_id][best_gt_idx][1]-1)
                predictions.append(pred_class)
                conf_scores.append(pred_score)
                matched.add(best_gt_idx)
                #del unmatched_gt[img_id][best_gt_idx] # remove, now that we've matched this box to a predicted box
            else: 
                # no ground truth bbox was matched with this prediction --> false positive
                ground_truths.append(-1)
                predictions.append(pred_class)
                conf_scores.append(pred_score)
            
        # any remaining have not been matched to a predicted bbox --> false negative
        for i, (_, true_cls) in enumerate(ground_truth_dict[img_id]):
            if i in matched: continue
            ground_truths.append(true_cls-1)
            predictions.append(-1)
            conf_scores.append(0.0)


    print("Finished generating true and predicted bounding boxes and classes.")
    return ground_truths, predictions, conf_scores

def plot_confusion_matrix(y_true, y_pred, coco_categories, output_dir):
    print("Plotting confusion matrix...")
    num_classes = len(coco_categories)
    class_ids = [cat["id"]-1 for cat in coco_categories] + [-1]
    class_names = [cat["name"] for cat in coco_categories]
    class_names_pred = class_names + ["no object\n(missed object)"]
    class_names_true = class_names + ["no object\n(ghost prediction)"]

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise

    # Create a mask: True for the last row & column (bottom-right cell)
    mask = np.zeros_like(cm, dtype=bool)
    mask[-1, -1] = True  # Hide bottom-right cell (No Object vs. No Object)
    
    # plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=class_names_pred, yticklabels=class_names_true, mask=mask)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir+"confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix to {output_dir}confusion_matrix.png")

def plot_pr_curve_per_class(y_true, y_pred, y_scores, coco_categories, output_dir):
    class_labels = [cat["name"] for cat in coco_categories]# + ["no object"]
    # class_ids = list(range(len(coco_categories))) + [-1]
    num_classes = len(class_labels)

    # For each class, compute precision-recall curve and plot
    rows = (num_classes // 3) + (num_classes % 3 > 0)  # Adjust rows based on the number of classes
    cols = 3  # Set number of columns to 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    for class_id, label in enumerate(class_labels):
        # Binarize the labels for each class
        y_true_cls = np.array([1 if y == class_id else 0 for y in y_true])
        y_scores_cls = np.array([score if pred == class_id else 0 for pred, score in zip(y_pred, y_scores)])
        
        # Only compute the precision-recall curve if there are positive samples
        # if any(y_true_class):  # Only proceed if there are positive samples for this class
        precision, recall, thresholds = precision_recall_curve(y_true_cls, y_scores_cls)
        auc_score = auc(recall, precision)
        
        ax = axes[class_id]
        ax.plot(recall, precision, label=f"PR Curve (AUC = {auc_score:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label}")

        best_threshold, best_threshold_index, best_f1 = threshold_and_idx_of_best_f1_from_pr(precision, recall, thresholds)
        ax.scatter(recall[best_threshold_index], precision[best_threshold_index], color='red', label=f'Best Thresh (F1 = {best_f1:.2f})')
        ax.annotate(f'Thresh: {best_threshold:.2f}', 
                        (recall[best_threshold_index], precision[best_threshold_index]), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', fontsize=8, color='red')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
    plt.tight_layout()
    #plt.legend()
    plt.savefig(output_dir+"pr_curve-per_class.png", bbox_inches='tight')
    plt.close()

def threshold_and_idx_of_best_f1_from_pr(precision, recall, thresholds):
    f1_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f1_scores.append(0)  # Avoid division by zero
        else:
            f1_scores.append(2 * (p * r) / (p + r))
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]
    return best_threshold, best_threshold_index, best_f1

def plot_pr_curve(y_true, y_pred, y_scores, coco_categories, output_dir):
    class_ids = [cat["id"]-1 for cat in coco_categories]# + [-1]

    # Initialize the predicted probabilities for the micro-average PR curve
    all_y_true = []
    all_y_scores = []

    # Compute per-class precision-recall curves
    precisions, recalls, ap_scores, weights = [], [], [], []
    for class_idx in class_ids:
        binary_y_true = np.array([1 if y == class_idx else 0 for y in y_true])
        binary_y_scores = np.array([score if pred == class_idx else 0 for pred, score in zip(y_pred, y_scores)])# Scores for this class
        
        all_y_true.extend(binary_y_true)
        all_y_scores.extend(binary_y_scores)
        
        precision, recall, _ = precision_recall_curve(binary_y_true, binary_y_scores)
        ap = average_precision_score(binary_y_true, binary_y_scores)
        
        precisions.append(precision)
        recalls.append(recall)
        ap_scores.append(ap)
        weights.append(sum(binary_y_true))  # Weight based on number of samples per class


    micro_precision, micro_recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    micro_ap = average_precision_score(all_y_true, all_y_scores)
    # Interpolate all PR curves at fixed recall points
    interp_recalls = np.linspace(0, 1, num=100)
    interp_precisions = [np.interp(interp_recalls, r[::-1], p[::-1]) for r, p in zip(recalls, precisions)]
    # Compute macro and weighted PR curves
    macro_precision = np.mean(interp_precisions, axis=0)
    macro_ap = np.mean(ap_scores)

    weighted_precision = np.average(interp_precisions, axis=0, weights=weights)
    weighted_ap = np.average(ap_scores, weights=weights)
 
    plt.plot(micro_recall, micro_precision, label=f'Micro-Averaged (AP = {micro_ap:.2f})', linewidth=2)
    plt.plot(interp_recalls, macro_precision, label=f'Macro-Averaged (AP = {macro_ap:.2f})', linewidth=2, linestyle="-.")
    plt.plot(interp_recalls, weighted_precision, label=f'Weighted-Averaged (AP = {weighted_ap:.2f})', linewidth=2, linestyle="dotted")


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro vs. Macro vs. Weighted PR Curves")
    plt.legend()
    plt.grid()
    plt.savefig(output_dir+"pr_curve.png", bbox_inches='tight')
    plt.close()
    
def main():
     # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    parser.add_argument('-cm', help="Confusion matrix", action='store_true')
    parser.add_argument('-pr', help="PR-curves", action='store_true')
    
    
    # Parse the arguments
    args = parser.parse_args()

    IMAGE_FOLDER = "/mnt/data0/martez/" + args.dataset_name + "/images/"   # Folder containing test images
    COCO_JSON = "/mnt/data0/martez/" + args.dataset_name + "/annotations/" + args.dataset_name + "_test.json"
    OUTPUT_DIR = "results/"
    PREDS_FILE = "results/predictions.json"

    coco_json = cc.load_from_file(COCO_JSON)
    pred_dict = cc.load_from_file(PREDS_FILE)

    if args.cm:
        print("Fetching ground truths...")
        ground_truth_dict = cc.load_image_to_bbox_and_cat_pairs_from_annotations(coco_json=coco_json)
        print("Matching predictions and ground truths...")
        ground_truth_classes, predicted_classes, scores = match_detections(ground_truth_dict=ground_truth_dict, pred_dict=pred_dict, conf_thresh=0.3)
        plot_confusion_matrix(y_true=ground_truth_classes, y_pred=predicted_classes, coco_categories=coco_json["categories"], output_dir=OUTPUT_DIR)
    if args.pr:
        print("Fetching ground truths...")
        ground_truth_dict = cc.load_image_to_bbox_and_cat_pairs_from_annotations(coco_json=coco_json)
        print("Matching predictions and ground truths...")
        ground_truth_classes, predicted_classes, scores = match_detections(ground_truth_dict=ground_truth_dict, pred_dict=pred_dict, conf_thresh=0.3)
        print("Plotting PR curves per class...")
        plot_pr_curve_per_class(y_true=ground_truth_classes, y_pred=predicted_classes, y_scores=scores, coco_categories=coco_json["categories"], output_dir=OUTPUT_DIR)
        print("Plotting average PR curve...")
        plot_pr_curve(y_true=ground_truth_classes, y_pred=predicted_classes, y_scores=scores, coco_categories=coco_json["categories"], output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()