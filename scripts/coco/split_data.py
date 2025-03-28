import argparse
import json
import random
import os
from sklearn.model_selection import train_test_split
from collections import Counter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils
import utils.pitfall_cameras_utils as pc
import utils.coco as ccu

def image_ids_with_annotations(images, annotations):
    # Filter out images with annotations
    image_ids_with_annotations = [img for img in images if img["id"] in [ann["image_id"] for ann in annotations]]
    image_ids_without_annotations = [img for img in images if img["id"] not in [ann["image_id"] for ann in annotations]]   
    return image_ids_with_annotations, image_ids_without_annotations

def gen_image_to_detected_classes_dict(images, annotations):
    print("Collecting image ids and detected class labels from annotations...")
    image_to_detected_classes = {}
    total_anns = len(annotations)
    for idx, ann in enumerate(annotations, start=1):
        image_id = ann["image_id"]
        if image_id not in image_to_detected_classes.keys():
            image_to_detected_classes[image_id] = []
        image_to_detected_classes[image_id].append(ann["category_id"])
        if idx % 5000 == 0 or idx == total_anns:
            print(f"-- Counted annotations for each image ID for {idx} out of {total_anns} annotations.", end="\r")
    return image_to_detected_classes

def filter_annotations(all_annotations, image_ids_to_filter):
    return [ann for ann in all_annotations if ann["image_id"] in image_ids_to_filter]

def filter_images(all_images, image_ids_to_filter):
    return [img for img in all_images if img["id"] in image_ids_to_filter]

def split_data(coco):
    """

    uses sklearn's stratified split function to perform the split, which takes a list of sample ids and 
    a list of their corresponding *values* that you want to stratify the split based on.

    """
    annotations = coco["annotations"]
    images = coco["images"]

    print("Finding positive and negative sample ids...")
    positive_sample_ids, negative_sample_ids = ccu.get_positive_and_negative_sample_ids_lists(images=coco["images"], annotations=coco["annotations"])
    ##### Split the positive data samples according to the class distribution found above
    # Collect all image IDs and the corresponding class labels for each image. image_id --> list of class labels in that image
    print("Finding the detected classes for each image...")
    image_to_detected_classes = gen_image_to_detected_classes_dict(images=images, annotations=annotations)
    # for each image id, note the most frequent class (max(...) operation on the set of detected classes 
    # with their counts as key/sorting criterion)
    def most_detected_class(img_id): 
        return max(set(image_to_detected_classes[img_id]), key=image_to_detected_classes[img_id].count)
    most_detected_class_per_image = {img_id: most_detected_class(img_id=img_id) for img_id in positive_sample_ids}
    class_counts = Counter(most_detected_class_per_image.values())
    # sort out the really rare classes (which can't handle a 80/10/10 split)
    RARE_THRESHOLD = 4
    rare_classes = {cls for cls, count in class_counts.items() if count <= RARE_THRESHOLD}
    normal_image_ids = [img_id for img_id in positive_sample_ids if most_detected_class_per_image[img_id] not in rare_classes]
    rare_image_ids = [img_id for img_id in positive_sample_ids if most_detected_class_per_image[img_id] in rare_classes]

    print("Splitting positive data samples... ")
    # split images 80/10/10 (except rare classes)
    # first split into train-(val+test), then evenly split (val+test) to val-test
    normal_classes_per_image = [most_detected_class_per_image[img_id] for img_id in normal_image_ids]
    train_norm, val_test_norm = train_test_split(normal_image_ids, train_size=0.7, stratify=normal_classes_per_image, random_state=42)
    val_classes_norm = [most_detected_class_per_image[img_id] for img_id in val_test_norm]
    val_norm, test_norm = train_test_split(val_test_norm, test_size=0.5, stratify=val_classes_norm, random_state=42)
    if rare_image_ids:
        # split rare images 50/25/25
        rare_classes_per_image = [most_detected_class_per_image[img_id] for img_id in rare_image_ids]
        train_rare, val_test_rare = train_test_split(rare_image_ids, train_size=0.5, stratify=rare_classes_per_image, random_state=42)
        val_classes_rare = [most_detected_class_per_image[img_id] for img_id in val_test_rare]
        val_rare, test_rare = train_test_split(val_test_rare, test_size=0.5, stratify=val_classes_rare, random_state=42)
    else: train_rare, test_rare, val_rare = [], [], []
    # combine
    train_pos_image_ids = train_norm + train_rare
    val_pos_image_ids = val_norm + val_rare
    test_pos_image_ids = test_norm + test_rare

    ##### Split the negative data samples ensuring the same positive-negative sample ratio
    print("Splitting negative data samples...")
    num_train_neg = int(len(train_pos_image_ids) * (len(negative_sample_ids) / len(positive_sample_ids)))
    num_neg_in_val_and_test = (len(negative_sample_ids) - num_train_neg)/2
    if num_neg_in_val_and_test % 2 == 0:
        num_val_neg = num_neg_in_val_and_test #int(len(val_pos_image_ids) * (len(negative_sample_ids) / len(positive_sample_ids)))
        num_test_neg = num_neg_in_val_and_test #int(len(test_pos_image_ids) * (len(negative_sample_ids) / len(positive_sample_ids)))
    else:
        num_val_neg = int(num_neg_in_val_and_test)
        num_test_neg = int(num_neg_in_val_and_test) + 1 
    
    train_neg_image_ids = random.sample(negative_sample_ids, num_train_neg)
    val_neg_image_ids = random.sample(list(set(negative_sample_ids) - set(train_neg_image_ids)), num_val_neg)
    test_neg_image_ids = list(set(negative_sample_ids) - set(train_neg_image_ids) - set(val_neg_image_ids))[:num_test_neg]

    print("Concatenating positive and negative samples and saving to files...")
    ##### Concatenate negative and positive samples
    train_ids = train_pos_image_ids + train_neg_image_ids
    val_ids = val_pos_image_ids + val_neg_image_ids
    test_ids = test_pos_image_ids + test_neg_image_ids

    ##### Create the coco object for each partition (same as the original coco, but with the ids filtered out)
    def gen_coco_partition(partition_name, ids):
        original_info = coco["info"]
        return {   "info": f"{partition_name.upper()}-partition for: {original_info}", 
                    "license": coco["license"],
                    "images": filter_images(all_images=images, image_ids_to_filter=ids),
                    "annotations": filter_annotations(all_annotations=annotations, image_ids_to_filter=ids),
                    "categories": coco["categories"]
                }
    train = gen_coco_partition("training", train_ids)
    val = gen_coco_partition("validation", val_ids)
    test = gen_coco_partition("test", test_ids)

    return train, val, test

def split_data_from_json(path):
    with open(path, "r") as f:
        coco = json.load(f)
    train, val, test = split_data(coco=coco)
    return train, val, test

def save_partitions(train, val, test, path):
    for name,obj in zip(["train", "val", "test"], [train, val, test]):
        utils.save_json_to_file(
            json_obj=obj,
            path=path+"_"+name+".json"
        )

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Perform stratefied split of provided COCO dataset.")
    parser.add_argument("dataset_name")

    # Parse the arguments
    args = parser.parse_args()
    dest_dir = "annotations/"+args.dataset_name
    src_file = dest_dir+"/info/"+args.dataset_name+"_all.json"
    dest_path_with_prefix = os.path.join(dest_dir, args.dataset_name)

    # Partition and save
    train, val, test = split_data_from_json(path=src_file)
    save_partitions(train=train, val=val, test=test, path=dest_path_with_prefix)
    print("Done! Phew. 😮‍💨")

    

if __name__ == "__main__":
    main()