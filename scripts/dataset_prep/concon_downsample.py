import json
import os
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils 
from coco import enforce_negratio as nr
from utils import coco as ccu

def main():
    target_no_of_images = 30000
    RARE_THRESHOLD = 2000 # only downsample images that have less than 2000 samples already 

    root = "annotations/controlled-conditions/info/"
    src_file = root+"controlled-conditions_all.json"
    stats_file = root+"statistics/statistics.json"
    
    print("Loading in files and determining the rare classes...")
    coco = utils.load_json_from_file(src_file)
    stats = utils.load_json_from_file(stats_file)["overall"]["class distribution"]
    annotations = coco["annotations"]
    images = coco["images"]
    rare_classes = [clss for clss in list(stats.keys()) if stats[clss] <= RARE_THRESHOLD] # list(filter(lambda clss: stats[clss] <= RARE_THRESHOLD, list(stats.keys())))
    print(f"The rare classes (fewer than {RARE_THRESHOLD} instances) are : {rare_classes}")

    print("Finding positive and negative sample ids...")
    positive_sample_ids, _ = ccu.get_positive_and_negative_sample_ids_lists(images=coco["images"], annotations=coco["annotations"])
    
    # Collect all image IDs and the corresponding class labels for each image. image_id --> list of class labels in that image
    print("Finding the detected classes for each image...")
    image_to_detected_classes = gen_image_to_detected_classes_dict(images=images, annotations=annotations)
    
    # Image ids that have a rare class (we will KEEP these)
    rare_image_ids = [img_id for img_id in positive_sample_ids if (any(clss in rare_classes for clss in image_to_detect_classes[img_id]))] # keep all of these
    # rare_image_ids = [img_id for img_id in positive_sample_ids if most_detected_class_per_image[img_id] in rare_classes] # keep all of these
    
    # group the remaining images by most detected class (non-rare)
    def most_detected_class(img_id): 
        return max(set(image_to_detected_classes[img_id]), key=image_to_detected_classes[img_id].count)
    non_rare_class_to_image_ids = {} 
    for img_id in positive_sample_ids:
        if img_id not in rare_image_ids:
            non_rare_class_to_image_ids[most_detected_class(img_id=img_id)].append(img_id)
    
    positive_samples_to_keep = {}
    # sample RARE_THRESHOLD no. of images per the non-rare class
    for ids in non_rare_class_to_image_ids.values():
        positive_samples_to_keep.update(random.sample(ids, RARE_THRESHOLD))
    # keep all the ids for the rare images
    positive_samples_to_keep.update(rare_image_ids)

    # filter the coco object wrt. these positive sample ids and reinforce neg ratio 
    coco["images"] = ccu.filter_images(all_images=coco["images"], image_ids_to_filter=list(positive_samples_to_keep))
    balanced_images = nr.get_balanced_images_wrt_target_neg_ratio(target_ratio=40, coco_data=coco)
    balanced_anns = nr.remove_annotations_for_removed_images(images=balanced_images, annotations=coco_data["annotations"])
    
    # save the new file
    coco["images"] = balanced_images
    coco["annotations"] = balanced_anns
    utils.save_json_to_file(json_obj=coco, path=root+"/controlled-conditions_downsample"+str(RARE_THRESHOLD)+".json") 

if __name__ == "__main__":
    main()