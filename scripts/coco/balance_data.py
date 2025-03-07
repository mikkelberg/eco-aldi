import json
import os
import argparse
import random
from collections import Counter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils



def get_category_id_from_name(category_name, categories):
    for cat in categories:
        if cat["name"] == category_name: return cat["id"]
    return None

def set_rare_categories_to_unknown(coco_data):
    category_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]} # category name -> category id
    unknown_id = category_name_to_id["unknown"]

    category_count = Counter() # occurrences of each category, Counter({id: count, id: count, id: count, ...})
    for ann in coco_data["annotations"]:
        category_count[ann["category_id"]] += 1
    
    categories_to_set_to_unknown = set()
    for cat_id in category_count:
        if category_count[cat_id] < 40:
            categories_to_set_to_unknown.add(cat_id)
    
    # set categories to unknown
    for ann in coco_data["annotations"]:
        if ann["category_id"] in categories_to_set_to_unknown:
            ann["category_id"] = unknown_id

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

def remove_annotations_with_unknown_images(images, annotations):
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

def enforce_negative_sample_ratio_in_dir(target_ratio: int, src_dir, dest_dir):
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith((".json")))]
    total_files = len(files)
    for index, filename in enumerate(files, start=1):
        json_path = os.path.join(src_dir, filename)
        coco_data = utils.load_json_from_file(json_path)
        
        balanced_images = get_balanced_images_wrt_target_neg_ratio(target_ratio=target_ratio, coco_data=coco_data)
        balanced_anns = remove_annotations_with_unknown_images(images=balanced_images, annotations=coco_data["annotations"])
        # Save the filtered data
        balanced_coco = {
            "info": f"Balanced version (40 % negative samples, and rare categories set to \"unknown\") of: {coco_data["info"]}",
            "license": coco_data["license"],
            "images": balanced_images,
            "annotations": balanced_anns,
            "categories": coco_data["categories"]
            }

        set_rare_categories_to_unknown(coco_data=balanced_coco)

        with open(os.path.join(dest_dir, filename), "w") as f:
            json.dump(balanced_coco, f, indent=4)
        
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files.")


  
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()
    
    src_dir = "annotations/"+args.dataset_name
    dest_dir = src_dir # overwrite
    enforce_negative_sample_ratio_in_dir(target_ratio=40, src_dir=src_dir, dest_dir=dest_dir)

if __name__ == "__main__":
    main()