import json
import os
import argparse
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.coco as ccu
from utils import utils

def get_positive_and_negative_samples(images, annotations):
    positive_sample_ids = {ann["image_id"] for ann in annotations}
    positive_samples = []
    negative_samples = []
    for img in images:
        if img["id"] in positive_sample_ids:
            positive_samples.append(img)
        else:
            negative_samples.append(img)
    return positive_samples, negative_samples

def remove_unknown_images_from_annotations(images, annotations):
    image_ids = {img["id"] for img in images}
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in image_ids]
    return filtered_annotations



def get_balanced_images_wrt_target_neg_ratio(target_ratio: int, coco_data: json):
    positive_samples, negative_samples = get_positive_and_negative_samples(images=coco_data["images"], annotations=coco_data["annotations"])

    no_of_negative_samples = len(negative_samples)
    no_of_positive_samples = len(positive_samples)
    no_of_negative_samples_for_target_ratio = int(no_of_positive_samples/(100-target_ratio)*target_ratio)
    no_of_negative_samples_to_keep = min(no_of_negative_samples, no_of_negative_samples_for_target_ratio) # because if e.g. target ratio is 0.25, we want to keep 1 negative sample for every 4 positives, so N_keep = target_ratio * no_of_positive_samples
    
    negative_samples_to_keep = random.sample(negative_samples, no_of_negative_samples_to_keep)
    return positive_samples + negative_samples_to_keep


def balance_subdataset_sizes_by_removing_negative_samples(target_max_size: int, src_dir, dest_dir):
    print()

def enforce_negative_sample_ratio_in_dir(target_ratio: int, src_dir, dest_dir):
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith((".json")))]
    total_files = len(files)
    for index, filename in enumerate(files, start=1):
        json_path = os.path.join(src_dir, filename)
        coco_data = utils.load_json_from_file(json_path)
        
        balanced_images = get_balanced_images_wrt_target_neg_ratio(target_ratio=target_ratio, coco_data=coco_data)
        balanced_anns = remove_unknown_images_from_annotations(images=balanced_images, annotations=coco_data["annotations"])
        # Save the filtered data
        balanced_coco = {
            "info": f"Balanced version (40 % negative samples) of: {coco_data["info"]}",
            "license": coco_data["license"],
            "images": balanced_images,
            "annotations": balanced_anns,
            "categories": coco_data["categories"]
            }

        with open(os.path.join(dest_dir, filename), "w") as f:
            json.dump(balanced_coco, f, indent=4)
        
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files.")


  
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("src_dir", nargs="?", help="", default="data-annotations/pitfall-cameras/merged-by-location_grouped-categories")
    
    # Parse the arguments
    args = parser.parse_args()
    
    enforce_negative_sample_ratio_in_dir(target_ratio=40, src_dir=args.src_dir, dest_dir="data-annotations/pitfall-cameras/balanced_neg_ratio_04_grouped-categories")

if __name__ == "__main__":
    main()