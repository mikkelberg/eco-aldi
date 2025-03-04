import argparse
import json
import random
import os
from sklearn.model_selection import train_test_split
import utils.utils as utils

def split_data(coco):
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
            path=os.path.join(path,"_",name,".json")
        )

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract categories from all VGG-CSV annotations into a single JSON-object (\"categories\") compatible with the COCO-format.")
    parser.add_argument("src_dir", nargs="?", help="Source directory containing the csv files.", default="data-annotations/pitfall-cameras/pitfall-cameras_all.json")
    parser.add_argument("dest_dir", nargs="?", help="Directory at which to save the generated categories.json.", default="data-annotations/pitfall-cameras/")
    parser.add_argument("file_prefix", nargs="?", help="Prefix for all the generated json files (<prefix>_test.json etc.)", default="pitfall-cameras")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Partition and save
    train, val, test = split_data_from_json(path=args.src_dir)
    dest_path_with_prefix = os.path.join(args.dest_dir, args.file_prefix)
    save_partitions(train=train, val=val, test=test, path=dest_path_with_prefix)

    

if __name__ == "__main__":
    main()