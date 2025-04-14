import json
import os
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils 

def main():
    target_no_of_images = 30000
    RARE_THRESHOLD = 2000 # only downsample images that have less than 2000 samples already 

    root = "annotations/controlled-conditions/info/"
    src_file = root+"controlled-conditions_all.json"
    stats_file = root+"statistics/statistics.json"
    
    coco = utils.load_json_from_file(src_file)
    stats = utils.load_json_from_file(stats_file)["overall"]["class distribution"]
    annotations = coco["annotations"]
    images = coco["images"]
    rare_classes = {clss for clss in stats.keys() if stats[clss] <= RARE_THRESHOLD}
    
    print("Finding positive and negative sample ids...")
    positive_sample_ids, _ = ccu.get_positive_and_negative_sample_ids_lists(images=coco["images"], annotations=coco["annotations"])
    
    # Collect all image IDs and the corresponding class labels for each image. image_id --> list of class labels in that image
    print("Finding the detected classes for each image...")
    image_to_detected_classes = gen_image_to_detected_classes_dict(images=images, annotations=annotations)
    
    rare_image_ids = [img_id for img_id in positive_sample_ids if (any(clss in rare_classes for clss in image_to_detect_classes[img_id]))] # keep all of these
    # rare_image_ids = [img_id for img_id in positive_sample_ids if most_detected_class_per_image[img_id] in rare_classes] # keep all of these
    
    def most_detected_class(img_id): 
        return max(set(image_to_detected_classes[img_id]), key=image_to_detected_classes[img_id].count)
    non_rare_class_to_image_ids = {clss: [] in stats.keys() if clss not in rare_classes}
    for img_id in positive_sample_ids:
        if img_id not in rare_image_ids:
            non_rare_class_to_image_ids[most_detected_class(img_id=img_id)].append(img_id)

    positive_samples_to_keep = []
    for ids in non_rare_class_to_image_ids.values()
        positive_samples_to_keep = random.sample(ids, RARE_THRESHOLD)
    positive_samples_to_keep.append(rare_image_ids)

    #loop through class to img_ids. if rare, append alle, if not rare, append sample.

    # reinforce negratio

if __name__ == "__main__":
    main()