import json
import os
import argparse
from collections import Counter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pitfall_cameras_utils as pc
from utils import utils

def update_global_stats(global_stats, stats_to_add):
    updated = global_stats.copy()
    updated["total images"] += stats_to_add["total images"]
    updated["total annotations"] += stats_to_add["total annotations"]
    updated["positive samples"] +=  stats_to_add["positive samples"]
    updated["negative samples"] += stats_to_add["negative samples"]
    for cls, count in stats_to_add.get("class distribution", {}).items():
        if cls in updated["class distribution"]: updated["class distribution"][cls] += count
        else: updated["class distribution"][cls] = count
    return updated

def get_coco_statistics(coco):
    """
    Extracts statistics from a COCO JSON dataset.
    
    :param coco_data: Loaded COCO JSON as a dictionary.
    :return: A dictionary with dataset statistics.
    """
    image_count = len(coco["images"])  
    annotation_count = len(coco["annotations"]) 
    category_count = len(coco["categories"])
    category_to_occurrence = Counter() # occurrences of each category, Counter({id: count, id: count, id: count, ...})
    image_to_ann_count = Counter()  # image id -> annotation count

    for ann in coco["annotations"]:
        image_to_ann_count[ann["image_id"]] += 1
        category_to_occurrence[ann["category_id"]] += 1
        
    positive_samples = sum(1 for count in image_to_ann_count.values() if count > 0)
    negative_samples = image_count - positive_samples

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]} # category id -> category name
    class_distribution_unsorted = {category_id_to_name[cid]: count for cid, count in category_to_occurrence.items()}
    class_distribution = dict(sorted(class_distribution_unsorted.items()))

    return {
        "total images": image_count,
        "total annotations": annotation_count,
        "no of categories": category_count,
        "positive samples": positive_samples,
        "negative samples": negative_samples,
        "class distribution": class_distribution
    }

def collect_statistics_from_directory(json_dir):
    """
    Collects statistics across multiple COCO JSON files.
    
    :param json_dir: Path to the directory containing COCO JSON files.
    """
    global_stats = {}
    per_file_stats = {} 
    
    files = [f for f in os.listdir(json_dir) if (os.path.isfile(os.path.join(json_dir, f)) and f.lower().endswith((".json")))]
    total_files = len(files)
    for index, filename in enumerate(files, start=1):
        json_path = os.path.join(json_dir, filename)
        coco_data = utils.load_json_from_file(json_path)
        this_files_stats = get_coco_statistics(coco_data)
        clean_filename = pc.get_img_folder_name_from_image_filename(filename.split(".")[0]) # remove .json)
        per_file_stats[ clean_filename] = this_files_stats

        if index == 1: global_stats = this_files_stats.copy()
        else: global_stats = update_global_stats(global_stats=global_stats, stats_to_add=this_files_stats)
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files.")

    print(json.dumps(global_stats, indent=4))
    return {"overall": global_stats, "per file": per_file_stats}

ann_dir = "data-annotations/pitfall-cameras/balanced_unknownfix_grouping"  


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()
    ann_dir = "annotations/"+args.dataset_name+"/"
    dest_path = "annotations/"+args.dataset_name+"/info/statistics/statistics.json"
    stats = collect_statistics_from_directory(ann_dir)
    with open(dest_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
