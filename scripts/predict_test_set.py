import utils.detectron_model as model
import argparse
import utils.coco as cc

def main():
     # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    parser.add_argument("config_file", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    IMAGE_FOLDER = "/mnt/data0/martez/" + args.dataset_name + "/images/"   # Folder containing test images
    COCO_JSON = "/mnt/data0/martez/" + args.dataset_name + "/annotations/" + args.dataset_name + "_test.json"
    PREDS_FILE = "results/predictions.json"
    MODEL_WEIGHTS = "output/training/" + args.dataset_name + "_val_model_best.pth"
    CONFIG_PATH = args.config_file

    print(f"Loading in model {CONFIG_PATH} with weights from {MODEL_WEIGHTS}...")
    predictor = model.load_model(CONFIG_PATH, MODEL_WEIGHTS, score_thresh=0.0)
    coco_json = cc.load_from_file(COCO_JSON)

    print(f"Generating predictions for images from {COCO_JSON}...")
    no_of_images = len(coco_json["images"])
    for idx, img in enumerate(coco_json["images"], start=1):
        img_path = f"{IMAGE_FOLDER}/{img["file_name"]}"
        pred_boxes, pred_classes, pred_scores = model.predict_bbox_and_class_with_score(predictor=predictor, image_path=img_path)
        model.save_predictions_to_file(image_id=img["id"], pred_boxes=pred_boxes, pred_classes=pred_classes, scores=pred_scores, dest=PREDS_FILE)
        if idx % 100 == 0 or idx == no_of_images:
            print(f"Generated predictions for {idx} out of {no_of_images} images.")

    print("Done!")

if __name__ == "__main__":
    main()