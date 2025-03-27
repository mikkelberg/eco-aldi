import numpy as np
import json
import pandas as pd
import argparse
import os
import re

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pitfall_cameras_utils as pc
import utils.controlled_conditions_utils as concon
import utils.utils as utils

def normalise_category_name(name:str):
    """ Normalise the string with the category name st. it's lower case and uses space separation, "unknown" or "(larvae)" comes last and typos (manually defined) are fixed."""
    name = name.lower()  # Convert to lowercase
    name = re.sub(r"[._]", " ", name)  # Replace dots and underscores with spaces
    name = re.sub(r"\s+", " ", name).strip()  # Remove extra spaces
    name = reorder_unknown(name)
    name = normalise_larvae(name)
    
    mother = utils.load_json_from_file(MOTHER_FILE)
    name_corrections = mother["name_corrections"]
    if name in name_corrections: # overwrite with the correction if it's there!
        name = name_corrections[name]
    return name

def reorder_unknown(name):
    """Moves 'unknown' to be after the first word, if present."""
    words = name.split()
    if "unknown" in words and words[-1] != "unknown":
        words.remove("unknown")
        words.append("unknown")  # Place "unknown" last
    return " ".join(words)   

def normalise_larvae(name):
    """Spells it 'larvae' and puts it in the end and in parentheses, e.g. carabidae (larvae)."""
    words = name.split()
    if "larva" not in words and "larvae" not in words:
        return name
    if "larva" in words:
        words.remove("larva")
    if "larvae" in words:
        words.remove("larvae")
    words.append("(larvae)")
    
    return " ".join(words)  

def clean_categories(cats):
    """Normalises all category names and merges according to redundancies (manually defined in the categories file)."""
    mother = utils.load_json_from_file(MOTHER_FILE)
    cats_to_remove = mother["remove"].keys()
    name_to_official_category = {}
    for off_cat, info in mother["categories"].items():
        for n in info["contains"]:
            name_to_official_category[n] = off_cat

    cleaned_set = set()
    removed = set()
    for category in cats:
        category_name = normalise_category_name(category) # lower case, space separation, "unknown" comes last

        if category_name in cats_to_remove:
            removed.add(category_name)
            continue
        category_name = name_to_official_category[category_name] # set to name groups

        cleaned_set.add(category_name) # insert the (corrected) category 

    return cleaned_set, removed

def create_coco_categories_from_set(cats:set):
    """Create category dictionary with unique, sorted category names"""
    return [
        {"id": idx + 1, "name": cat}
        for idx, cat in enumerate(sorted(cats))  # Sorted to ensure consistency
    ]

def save_categories_to_file(cats, og_cats, dest_dir, filename):
    path = os.path.join(dest_dir, filename)
    categories = {}
    categories["info"] = {
        "description": "The cleaned categories are matched between the source and target dataset (respectively controlled-conditions and pitfall-cameras), and the original_categories are all of the unique classifications from this dataset before any grouping for redundancies and other cleaning.",
        "number_of_original_categories": len(list(og_cats)),
        "number_of_official_categories": len(list(cats))}
    categories["cleaned_categories"] = create_coco_categories_from_set(cats)
    categories["original_categories"] = [{"name": cat} for cat in sorted(og_cats)]
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
        normalised_cat_name = normalise_category_name(name=cat)
        unique_categories.add(normalised_cat_name)
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
    print(f"Extracted {len(list(categories))} unique categories from the dataset.")
    
    return categories

def normalised_name_to_concon_code(categories:list[dict]):
    name_to_original_ids = {}
    for i, cat in enumerate(categories):
        normalised_name = normalise_category_name(name=cat["name"])
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
        name = normalise_category_name(name= code_to_insect[str(code)])
        concon_categories.append({"code": code, "name": name})
    # eliminate duplicates
    categories = set()
    name_to_concon_code = normalised_name_to_concon_code(concon_categories) # keep track of which ids map to the same species
    for n in sorted(list(name_to_concon_code.keys())): # add each unique name
        categories.add(n)
    print(f"Extracted {len(list(categories))} unique categories from the dataset.")
    return categories, ignored_codes


MOTHER_FILE = "annotations/categories.json"
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract categories from all VGG-CSV annotations into a single JSON-object (\"categories\") compatible with the COCO-format.")
    parser.add_argument("dataset_name", help="")
    
    # Parse the arguments
    args = parser.parse_args()

    root = "annotations/"

    if args.dataset_name == "pitfall-cameras":
        SRC_DIR = root+"pitfall-cameras/originals/"
        DEST_DIR = root+"pitfall-cameras/info/"
        
        categories_set = extract_categories_from_vgg_csv_dir(SRC_DIR)
        categories_set_clean, removed = clean_categories(categories_set)
        # coco_categories = create_coco_categories_from_set(categories_set_clean)
        print(f"Extracted {len(list(categories_set_clean))} after cleaning up/grouping.")
        print(f"Removed the following {len(list(removed))}: {list(removed)}")
        save_categories_to_file(cats=categories_set_clean, og_cats=categories_set, dest_dir=DEST_DIR, filename="pitfall-cameras_categories.json")
    elif args.dataset_name == "controlled-conditions":
        DEST_DIR = root+"controlled-conditions/info/"
        codes = concon.get_insect_codes_from_paper_conditions()
        
        og_categories, _ = extract_categories_from_controlled_conditions_metadata(codes=codes)
        categories, removed = clean_categories(cats=og_categories)
        print(f"Extracted {len(list(categories))} after cleaning up/grouping.")
        print(f"Removed the following {len(list(removed))}: {list(removed)}")
        save_categories_to_file(cats=categories, og_cats=og_categories, dest_dir=DEST_DIR, filename="controlled-conditions_categories.json")


if __name__ == "__main__":
    main()