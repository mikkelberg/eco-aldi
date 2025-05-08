import os
import argparse
import pandas as pd
from pandas import DataFrame

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils import coco
from coco.enforce_negratio import remove_annotations_for_removed_images

def clean_images(src_coco, meta: DataFrame):
    src_images = src_coco["images"]
    total_images = len(src_images)
    category_names = [cat["name"] for cat in src_coco["categories"]]
    cleaned_images = []
    for i,img in enumerate(src_images, start=1):
        date, cam = date_cam_from_filename(img["file_name"])
        category_name = meta[(meta['date'] == date) & (meta['camera']==cam)]["insect_name"]
        if not category_name.empty: # this data+cam is annotated+paper background (otherwise it would not be in the meta.csv)
            category_name = category_name.item()
            if category_name in category_names:
                cleaned_images.append(img)
        if i % 5000 == 0 or i == total_images: print(f"Sorted through {i} out of {total_images} images.", end="\r")
    
    print("\nFiltering annotations...")
    filtered_anns = remove_annotations_for_removed_images(images=cleaned_images, annotations=src_coco["annotations"])
    print("Done! The unwanted images have been removed from the coco-json file.")
    print(f"Ended with {len(cleaned_images)} images and {len(filtered_anns)} annotations.")
    src_coco["images"] = cleaned_images
    src_coco["annotations"] = filtered_anns

def date_cam_from_filename(filename):
    date, cam, _ = filename.split("/")
    return date, cam

def category_name_from_date_cam(date, cam, meta):
    return meta[(meta['date'] == date) & (meta['camera']==cam)]["insect_name"].item()

def add_categories_to_anns(src_coco, meta):
    print("Adding categories to the annotatiosn...")
    anns = src_coco["annotations"]
    img_id_to_filename = coco.image_id_to_filename(coco_images=src_coco["images"])
    cat_name_to_id = coco.category_name_to_id(coco_categories=src_coco["categories"])
    total_anns = len(anns)

    for i, ann in enumerate(anns, start=1):
        img_id = ann["image_id"]
        filename = img_id_to_filename[img_id]
        date, cam = date_cam_from_filename(filename=filename)
        category_name = category_name_from_date_cam(date=date, cam=cam, meta=meta)
        ann["category_id"] = cat_name_to_id[category_name]
        if i % 5000 == 0 or i == total_anns: print(f"Sorted through {i} out of {total_anns} annotations.", end="\r")
    print("\nDone!")
    
def main():
    # Set up command-line argument parsing
    # parser = argparse.ArgumentParser(description="Merge the json files for COCO-datasets in a directory.")
    src_file = "annotations/controlled-conditions/info/controlled-conditions_all_unfiltered.json"
    dest_path = "annotations/controlled-conditions/controlled-conditions_all.json"
    coco = utils.load_json_from_file(src_file)
    
    meta = "annotations/controlled-conditions/info/meta.csv"
    metadf = pd.read_csv(meta, dtype={"date": str})

    clean_images(src_coco=coco, meta=metadf)

    add_categories_to_anns(src_coco=coco, meta=metadf)

    utils.save_json_to_file(json_obj=coco, path=dest_path)
if __name__ == "__main__":
    main()