import json
import os
import argparse
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils 
from coco import enforce_negratio as nr
from coco.split_data import gen_image_to_detected_classes_dict
from utils import coco as ccu

def main():
    # target_no_of_images = 30000
    RARE_THRESHOLD = 2000 # only downsample images that have less than 2000 samples already 

    root = "annotations/controlled-conditions/"
    src_file = root+"controlled-conditions_train.json"
    stats_file = root+"info/statistics/statistics.json"
    
    print("Loading in files and determining the rare classes...")
    coco = utils.load_json_from_file(src_file)
    category_name_to_id = ccu.category_name_to_id(coco_categories=coco["categories"])
    stats = utils.load_json_from_file(stats_file)["per file"]["controlled-conditions_train"]["class distribution (images)"]
    annotations = coco["annotations"]
    images = coco["images"]
    rare_classes = [category_name_to_id[clss] for clss in list(stats.keys()) if stats[clss] < RARE_THRESHOLD] # list(filter(lambda clss: stats[clss] <= RARE_THRESHOLD, list(stats.keys())))
    print(f"The rare classes (fewer than {RARE_THRESHOLD} instances) are : {rare_classes}")

    print("Finding positive and negative sample ids...")
    positive_sample_ids, negative_sample_ids = ccu.get_positive_and_negative_sample_ids_lists(images=coco["images"], annotations=coco["annotations"])
    
    # Collect all image IDs and the corresponding class labels for each image. image_id --> list of class labels in that image
    print("Finding the detected classes for each image...")
    image_to_detected_classes = gen_image_to_detected_classes_dict(images=images, annotations=annotations)
    
    print("Sorting through positive samples, making note of which to keep and which to downsample")
    def most_detected_class(img_id): 
        return max(set(image_to_detected_classes[img_id]), key=image_to_detected_classes[img_id].count)
    # sort through positive samples, keeping all images with rare occurrences, and grouping the rest by most detected (non-rare class)
    # we also count the number of images that a category occurs in where it was NOT the most detected (so we can account for this number in the number to sample)
    non_rare_class_to_image_ids = {category_name_to_id[clss]: {"image_ids": [], "no_of_other_image_occurrences": 0} for clss in list(stats.keys()) if category_name_to_id[clss] not in rare_classes}
    rare_image_ids = set()
    for img_id in positive_sample_ids:
        classes = image_to_detected_classes[img_id]
        if (any(clss in classes for clss in rare_classes)): 
            # contains a rare class --> keep and note it for any non-rare classes in the image
            rare_image_ids.add(img_id)
            for other_clss in classes:
                if other_clss not in rare_classes: non_rare_class_to_image_ids[other_clss]["no_of_other_image_occurrences"] += 1
        else: 
            # no rare detections --> add to list for most detected and note for the other (non-rare) classes
            clss = most_detected_class(img_id)
            non_rare_class_to_image_ids[clss]["image_ids"].append(img_id)
            for other_clss in classes:
                if other_clss != clss: non_rare_class_to_image_ids[other_clss]["no_of_other_image_occurrences"] += 1
    
    positive_samples_to_keep = set()
    positive_samples_to_keep.update(rare_image_ids) # keep all the ids for the rare images
    for key,ids in non_rare_class_to_image_ids.items(): 
        # sample up to RARE_THRESHOLD no. of images per the non-rare class 
        no_to_sample = RARE_THRESHOLD-ids["no_of_other_image_occurrences"] # (account for (and keep) the ones with rare detections)
        positive_samples_to_keep.update(random.sample(ids["image_ids"], no_to_sample))
    
    # filter the coco object wrt. these positive sample ids and reinforce neg ratio 
    unbalanced_images = list(positive_samples_to_keep) + negative_sample_ids 
    coco["images"] = ccu.filter_images(all_images=coco["images"], image_ids_to_filter=unbalanced_images)
    balanced_images = nr.get_balanced_images_wrt_target_neg_ratio(target_ratio=40, coco_data=coco)
    balanced_anns = nr.remove_annotations_for_removed_images(images=balanced_images, annotations=coco["annotations"])
    
    # save the new file
    coco["images"] = balanced_images
    print(len(unbalanced_images))
    print(len(balanced_images))
    coco["annotations"] = balanced_anns
    utils.save_json_to_file(json_obj=coco, path=root+"/downsample"+str(RARE_THRESHOLD)+"/controlled-conditions_train-downsample"+str(RARE_THRESHOLD)+".json") 

if __name__ == "__main__":
    main()