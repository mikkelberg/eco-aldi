import os
import argparse
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

def clean_images(src_coco, meta):
    src_images = src_coco["images"]
    # loop through images
    # extract dat+cam
    # if not in meta.csv --> remove this image + annotations
    return 

def add_categories(src_coco, dest_path, categories, meta):
    new_coco = {}


    # loop through annotations
        # get img id
        # look up image's filename
        # extract date + camera from filename
        # look up name for rrthat data+camera
        # translate insect name to official name (grouped)
        # if should be removed --> remember this image_id
        # add category id for the insect name
    
    # remove images that were marked to remove

    # save new coco file

    utils.save_json_to_file(json_obj=new_coco, path=dest_path)
def main():
    # Set up command-line argument parsing
    # parser = argparse.ArgumentParser(description="Merge the json files for COCO-datasets in a directory.")
    src_file = "annotations/controlled-conditions/info/controlled-conditions_all_no-cats.json"
    dest_path = "annotations/controlled-conditions/info/controlled-conditions_all.json"
    src_coco = utils.load_json_from_file(src_file)

    categories_file =  "annotations/categories.json"
    categories_info = utils.load_json_from_file(categories_file)
    categories = [{"id": categories_info["categories"][cat]["id"], "name": cat} for cat in categories_info["categories"].keys()]
    
    meta = "annotations/controlled-conditions/info/meta.csv"
    metadf = pd.read_csv(meta)

    cleaned_coco = clean_images(src_coco=src_coco, meta=metadf)
    add_categories(src_coco=cleaned_coco, dest_path=dest_path, categories=categories, meta=metadf)
    
if __name__ == "__main__":
    main()