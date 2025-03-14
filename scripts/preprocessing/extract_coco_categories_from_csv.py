import numpy as np
import json
import pandas as pd
import sys
import argparse
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pitfall_cameras_utils as pc
import utils.controlled_conditions_utils as concon
import utils.utils as utils

def clean_categories(cats):
    """Normalises all category names and merges according to typos/redundancies (manually defined in the dict above)"""
    mother = utils.load_json_from_file(MOTHER_FILE)
    cats_to_remove = mother["remove"].keys()
    name_corrections = mother["name_corrections"]
    
    name_to_official_category = {}
    for off_cat, info in mother["categories"].items():
        for n in info["contains"]:
            name_to_official_category[n] = off_cat

    cleaned_set = set()
    removed = set()
    for category in cats:
        category_name = pc.normalise_category_name(category) # lower case, space separation, "unknown" comes last
        
        if category_name in name_corrections: # overwrite with the correction if it's there!
            category_name = name_corrections[category_name]

        if category_name in cats_to_remove:
            removed.add(category_name)
            continue
        category_name = name_to_official_category[category_name] # set to name groups

        cleaned_set.add(category_name) # insert the (corrected) category 

    return cleaned_set, removed

def create_coco_categories_from_set(cats:set):
    """Create category dictionary with unique, sorted category names"""
    return [
        {"id": idx + 1, "name": cat, "supercategory": "insect"}
        for idx, cat in enumerate(sorted(cats))  # Sorted to ensure consistency
    ]

def save_categories_to_file(cats, dest_dir, filename):
    path = os.path.join(dest_dir, filename)
    categories = {}
    categories["categories"] = cats
    with open(path, "w") as f:
        json.dump(categories, f, indent=4)

def extract_categories_from_vgg_csv(src):
    df = pd.read_csv(src, on_bad_lines='skip')
    unique_categories = set()  # Store unique category names
    for row in df.itertuples():
        if row.region_count <= 0:
            continue
        # if there is a detection, append the category name
        cat = pc.extract_category_name_from_region_attributes(row.region_attributes)

        # ignore the detection, if there is not associated label with the annotation
        no_insect_label_but_was_annotated = not bool(cat)
        if no_insect_label_but_was_annotated: 
            continue
        
        unique_categories.add(cat)
    return unique_categories

def extract_categories_from_vgg_csv_dir(src_dir):
    """Runs through a given directory of vgg-csv annotation files and extracts all unique categories (returned as a set)"""
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".csv")))]
    total_files = len(files)
    
    categories = set()
    
    for index, filename in enumerate(files, start=1): 
        # Build the full path to the file
        src_file_path = os.path.join(src_dir, filename)
        cats = extract_categories_from_vgg_csv(src_file_path)
        categories.update(cats)
        # Print progress every 5 files
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")
    return categories

def extract_clean_categories_from_vgg_csv_dir(src_dir):
    categories_set = extract_categories_from_vgg_csv_dir(src_dir)
    categories_set_clean, removed = clean_categories(categories_set)
    coco_categories = create_coco_categories_from_set(categories_set_clean)
    print(f"Extracted {len(coco_categories)} categories from the annotations.")
    print(f"Removed the following {len(list(removed))}: {list(removed)}")
    return coco_categories

def normalised_name_to_concon_code(categories:list[dict]):
    name_to_original_ids = {}
    for i, cat in enumerate(categories):
        normalised_name = pc.normalise_category_name(name=cat["name"])
        if normalised_name in name_to_original_ids.keys():
            name_to_original_ids[normalised_name].append(cat["code"])
        else:
            name_to_original_ids[normalised_name] = [cat["code"]]
    return name_to_original_ids
    

def extract_categories_from_controlled_conditions_metadata(codes:list[int]):
    code_to_insect = concon.get_code_to_insect_dict()
    ignored_codes = {}
    concon_categories = []
    for code in codes:
        if str(code) not in code_to_insect.keys():
            ignored_codes[code] = f"Insect code ({code}) denoted for these images did not exist."
            continue
        name = pc.normalise_category_name(name= code_to_insect[str(code)])
        concon_categories.append({"code": code, "name": name})
    # eliminate duplicates
    categories = set()
    name_to_concon_code = normalised_name_to_concon_code(concon_categories) # keep track of which ids map to the same species
    for n in sorted(list(name_to_concon_code.keys())): # add each unique name
        categories.add(n)
    print(f"Extracted {len(list(categories))} unique categories from the dataset.")
    return categories, ignored_codes

def remove_if_not_in_target_dataset(cats, target_dataset):
    target_categories = utils.load_json_from_file("annotations/"+target_dataset+"/info/categories.json")
    target_category_names = [cat["name"] for cat in target_categories["categories"]]
    
    approved_cats = set()
    removed_cats = set()
    for cat in cats:
        if cat in target_category_names:
            approved_cats.add(cat)
        else:
            removed_cats.add(cat)
    for unknown in target_categories["categories_set_to_unknown_due_to_low_frequency"]:
        if unknown in approved_cats:
            approved_cats.remove(unknown)
            removed_cats.add(unknown)
    missing_cats = set()
    for cat in target_category_names:
        if cat not in approved_cats and cat not in target_categories["categories_set_to_unknown_due_to_low_frequency"]:
            missing_cats.add(cat)
    
    return approved_cats, removed_cats, missing_cats

def extract_clean_categories_from_controlled_conditions_metadata(codes:list[int]):
    categories, _ = extract_categories_from_controlled_conditions_metadata(codes=codes)
    categories, removed = clean_categories(cats=categories)
    print(f"Extracted {len(list(categories))} after cleaning up/grouping.")
    print(f"Removed the following {len(list(removed))}: {list(removed)}")
    coco_categories = create_coco_categories_from_set(categories)
    
    return coco_categories


MOTHER_FILE = "annotations/categories.json"
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract categories from all VGG-CSV annotations into a single JSON-object (\"categories\") compatible with the COCO-format.")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    FILENAME = "categories.json"
    root = "annotations/"

    if args.dataset_name == "pitfall-cameras":
        SRC_DIR = root+"pitfall-cameras/originals/"
        DEST_DIR = root+"pitfall-cameras/info/"
        
        coco_categories = extract_clean_categories_from_vgg_csv_dir(SRC_DIR)
        save_categories_to_file(cats=coco_categories, dest_dir=DEST_DIR, filename=FILENAME)
    elif args.dataset_name == "controlled-conditions":
        DEST_DIR = root+"controlled-conditions/info/"
        codes = concon.get_insect_codes_from_paper_conditions()
        
        coco_categories = extract_clean_categories_from_controlled_conditions_metadata(codes=codes)
        save_categories_to_file(cats=coco_categories, dest_dir=DEST_DIR, filename=FILENAME)


if __name__ == "__main__":
    main()