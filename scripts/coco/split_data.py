import argparse
import json
import random
import os
from sklearn.model_selection import train_test_split

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

def gen_basic_coco(coco, partition:str):
    return {"info": f"{partition.upper()}-partition for: {coco["info"]}",
            "license": coco["license"],
            "images": [],
            "annotations": [],
            "categories": coco["categories"]
            }

def gen_image_to_detected_classes_dict(images, annotations):
    print("Collecting image ids and detected class labels from annotations...")
    image_to_detected_classes = {}
    total_anns = len(annotations)
    for idx, ann in enumerate(annotations, start=1):
        image_id = ann["image_id"]
        if image_id not in image_to_detected_classes.keys():
            image_to_detected_classes[image_id] = []
        image_to_detected_classes[image_id].append(ann["category_id"])
        if idx % 500 == 0 or idx == total_anns:
            print(f"-- Counted annotations for each image ID for {idx} out of {total_anns} annotations.")
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

    positive_sample_ids, negative_sample_ids = ccu.get_positive_and_negative_sample_ids_lists(images=coco["images"], annotations=coco["annotations"])
    ##### Split the positive data samples according to the class distribution found above
    # Collect all image IDs and the corresponding class labels for each image. image_id --> list of class labels in that image
    image_to_detected_classes = gen_image_to_detected_classes_dict(images=images, annotations=annotations)
    # for each image id, note the most frequent class (max(...) operation on the set of detected classes 
    # with their counts as key/sorting criterion)
    def most_detected_class(img_id): 
        return max(set(image_to_detected_classes[img_id]), key=image_to_detected_classes[img_id].count)
    most_detected_class_per_image = [most_detected_class(img_id=img_id) for img_id in positive_sample_ids]
    # first split into train-(val+test), then evenly split (val+test) to val-test - the result is a 80/10/10 split
    train_pos_image_ids, temp_ids = train_test_split(positive_sample_ids, test_size=0.3, stratify=most_detected_class_per_image, random_state=42)
    most_detected_class_per_image_for_val_and_test = [most_detected_class_per_image[positive_sample_ids.index(id)] for id in temp_ids] #
    val_pos_image_ids, test_pos_image_ids = train_test_split(temp_ids, test_size=0.5, stratify=most_detected_class_per_image_for_val_and_test, random_state=42)
    
    '''
    ##### Split the negative data samples based on the location (i.e. background) distribution
    # the location is defined in the prefix of the image's id
    negative_sample_image_ids = [img["id"] for img in images if img["id"] not in positive_sample_image_ids]
    locations_per_image = [pc.get_location_prefix_from_image_filename(image_id) for image_id in negative_sample_image_ids]
    negative_sample_train_image_ids, temp_ids = train_test_split(negative_sample_image_ids, test_size=0.3, stratify=locations_per_image, random_state=42)
    locations_per_image_for_val_and_test = [locations_per_image[negative_sample_image_ids.index(id)] for id in temp_ids]
    negative_sample_val_image_ids, negative_sample_test_image_ids = train_test_split(temp_ids, test_size=0.5, stratify=locations_per_image_for_val_and_test, random_state=42)
    '''

    ##### Split the negative data samples ensuring the same positive-negative sample ratio
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

    ##### Concatenate negative and positive samples
    train_ids = train_pos_image_ids + train_neg_image_ids
    val_ids = val_pos_image_ids + val_neg_image_ids
    test_ids = test_pos_image_ids + test_neg_image_ids

    ##### Create the coco object for each partition (same as the original coco, but with the ids filtered out)
    def gen_coco_partition(partition_name, ids):
        return {   "info": f"{partition_name.upper()}-partition for: {coco["info"]}", # FIXME not possible to write to dicts this way
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

    

if __name__ == "__main__":
    main()