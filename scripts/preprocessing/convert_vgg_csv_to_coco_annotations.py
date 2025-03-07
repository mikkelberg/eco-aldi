import json
import pandas as pd
import os
import re
import argparse

from extract_coco_categories_from_csv import extract_clean_categories_from_vgg_csv_dir, save_categories_to_file
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pitfall_cameras_utils as pc
from utils import utils

def get_filename_for_csv_annotations(date:str, camera, flash=False):
    date_list = date.split("-")
    date_list.reverse()
    src_date = "".join(date_list)
    flash = "on" if flash else "off"
    return f"{src_date} {camera} Flash {flash}_csv.csv"

def get_camera_from_csv_filename(csv_name):
    return csv_name.split(" ")[1]

def get_datetime_from_csv_filename(csv_name):
    og_date = csv_name.split(" ")[0]
    date = re.findall('..', og_date)
    date.reverse()
    date = "-".join(date)
    return date
    
def gen_dict_for_cam_and_date_to_img_folder_name():
    with open(pc.INFO_FILE_PATH, "r") as f:
        folders = json.load(f)["folders"]
    date_and_camera_to_folder_name = {}
    for folder in folders.keys():
        camera = folders[folder]["specs"]["camera"]
        dates = folders[folder]["specs"]["dates"]
        for date in dates:
            date_and_camera_to_folder_name[f"{camera}-{date}"] = folder
    return date_and_camera_to_folder_name

def load_json(file):
    o = {}
    with open(file, 'r') as f:
        o = json.load(f)
    return o

def gen_info(img_folder_name):
    info_json = load_json(pc.INFO_FILE_PATH)
    specs = info_json["folders"][img_folder_name]["specs"]
    field = specs["field"]
    crop = info_json["crops"][specs["crop"]]["name"]
    camera = specs["camera"]
    date_first = specs["dates"][0]
    date_last = specs["dates"][-1]

    info = {}
    description = f"{field} {crop} field, camera {camera} - recorded (and annotated) between 20{date_first} and 20{date_last} (specific date in filename)."
    info["description"] = description
    info["date_created"] = date_first
    return info

def image(row, images_dir):
    width, height = utils.read_image_size(images_dir+row.filename)

    image = {}
    image["id"] = row.fileid
    image["height"] = width 
    image["width"] = height 
    image["file_name"] = row.filename
    return image

def annotation(row, category_id):
    annotation = {}
    annotation["id"] = row.fileid + "_" + str(row.region_id) 
    annotation["image_id"] = row.fileid
    annotation["category_id"] = category_id
    annotation["segmentation"] = [] # NOTE blank? We don't have this. Wondering if this needs to be disabled if we use flat-bug data 
    shape = json.loads(row.region_shape_attributes)
    annotation["area"] = shape["width"]*shape["height"]
    annotation["bbox"] = [shape["x"], shape["y"], shape["width"], shape["height"]]
    annotation["iscrowd"] = 0
    return annotation

def record_ignored_images(ignored_images, dest_dir):
    already_ignored = load_json(pc.IGNORED_IMAGES_PATH)
    if not bool(already_ignored): already_ignored = {}
    with open(pc.IGNORED_IMAGES_PATH, 'w') as f:
        for key in ignored_images.keys():
            if key in already_ignored:
                already_ignored[key].append(ignored_images[key])
            else:
                already_ignored[key] = ignored_images[key]
        json.dump(already_ignored, f, indent=4)

def convert(file_prefix, img_folder_name, csv_file_path, coco_file_destination, categories_file, images_dir):
    data = pd.read_csv(csv_file_path, on_bad_lines='skip')

    data['fileid'] = file_prefix + data['filename'].astype(str) # create fileid column
                                                                # id is the file prefix prepended to the original filename (this corresponds to the correct filename in OUR system)
    data['filename'] = img_folder_name + "/" + data['fileid'].astype(str) # update the filename to have the entire path (and of course the correct filename, which we just saved in the fileid-column)
    
    images_to_clean_out = set()
    
    # Create categories entries 
    categories = load_json(categories_file)["categories"]
    
    # Create images entries, one for each image
    images = []
    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row, images_dir))
    
    # Create annotations entries
    annotations = []
    name_mappings = load_json(categories_file)["name_mappings"]
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    for row in data.itertuples():
        if row.region_count <= 0:
            continue
        # if there is a detection, append the annotation
        category_name = pc.extract_category_name_from_region_attributes(row.region_attributes)

        no_insect_label_but_was_annotated = not bool(category_name)
        if no_insect_label_but_was_annotated: 
            csv_name = csv_file_path.split("/")[-1]
            og_filename = row.fileid.split("_")[-1]
            if img_folder_name not in ignored_images.keys():
                ignored_images[img_folder_name] = [] 
            ignored_images[img_folder_name].append(pc.ignored_img(filename=row.fileid, explanation="Incomplete annotation: No insect class in annotation (region_attributes).", og_csv_name=csv_name, og_filename=og_filename))
            images_to_clean_out.add(row.fileid)
            continue

        category_name = pc.normalise_category_name(category_name)
        if category_name in name_mappings.keys():
            category_name = name_mappings[category_name] # make correction, if needed
        if category_name in pc.names_to_group:
            category_name = pc.names_to_group[category_name]
        if category_name in pc.categories_to_set_to_unknown:
            category_name = "unknown"
        category_id = category_name_to_id[category_name]
        ann = annotation(row,category_id)
        annotations.append(ann)
        if category_id == 4: print(row.filename)

    # Clean out images that need to be removed due to errors (undefined annotation)
    images = [img for img in images if img["id"] not in images_to_clean_out]
    annotations = [ann for ann in annotations if ann["image_id"] not in images_to_clean_out]
    

    # Create final coco-json file 
    data_coco = {}
    data_coco["info"] = gen_info(img_folder_name)
    data_coco["license"] = None
    data_coco["images"] = images
    data_coco["annotations"] = annotations
    data_coco["categories"] = categories
    with open(coco_file_destination, "w") as f:
        json.dump(data_coco, f, indent=4)

def convert_all_vgg_csv_in_dir_to_COCO(src_dir, dest_dir, cats_file, images_dir):
    """Runs through a given directory of vgg-csv annotation files and converts all of them to COCO JSON format."""
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".csv")))]
    total_files = len(files)
    
    cam_and_date_to_img_folder_name = gen_dict_for_cam_and_date_to_img_folder_name()
    
    for index, filename in enumerate(files, start=1): 
        # Build the full path to the file
        src_file_path = os.path.join(src_dir, filename)

        date = get_datetime_from_csv_filename(filename)
        img_folder_name = cam_and_date_to_img_folder_name[f"{get_camera_from_csv_filename(filename)}-{date}"]
        specs = pc.get_specs_from_info(img_folder_name)
        flash = filename.split(" ")[-1].startswith("on")
        file_prefix = pc.get_file_prefix_from_specs(field=specs["field"], crop=specs["crop"], camera=specs["camera"], date=date, flash=flash)
        
        convert(file_prefix=file_prefix, img_folder_name=img_folder_name, csv_file_path=src_file_path, coco_file_destination=f"{dest_dir}{file_prefix}.json", categories_file=cats_file, images_dir=images_dir)
        # Print progress every 5 files
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")
    
    print(f"Converted all the csv-annotations in {src_dir}")
   

ignored_images = {} # {"csv_file_name": [{"filename": "bla", "explanation": "bla", "original_filename": "bla", "original_csv_file" : "bla"}, ...], ...}

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert each VGG-CSV annotations in a directory to COCO JSON format.")
    parser.add_argument("dataset_name", help="E.g. pitfall-cameras (matches the directory)")
    parser.add_argument("images_dir", help="Path to images directory.")

    # python scripts/preprocessing/convert_vgg_csv_to_coco_annotations.py pitfall-cameras ../../../mnt/data0/martez/images/

    # Parse the arguments
    args = parser.parse_args()

    src_dir = "annotations/" + args.dataset_name + "/originals/"
    dest_dir = "annotations/" + args.dataset_name + "/originals-converted/"
    categories_file = "annotations/" + args.dataset_name + "/info/categories.json"
    ignored_images_path = "annotations/" + args.dataset_name + "/info/ignored_images.json"
    
    # Extract categories (ensures consistent category ids)
    coco_categories = extract_clean_categories_from_vgg_csv_dir(src_dir=src_dir)
    save_categories_to_file(
        cats=coco_categories, dest_dir="annotations/" + args.dataset_name + "/info/", filename="categories.json", 
        mappings=pc.name_mappings, groupings=pc.names_to_group, 
        unknown_overwrites=pc.categories_to_set_to_unknown)
    
    # Convert
    convert_all_vgg_csv_in_dir_to_COCO(src_dir=src_dir, dest_dir=dest_dir, cats_file=categories_file, images_dir=args.images_dir)

    # Store record of ignored images
    with open(ignored_images_path, 'w') as f:
        json.dump(ignored_images, f, indent=4)
    print(f"Ignored {sum([len(ignored_images[key]) for key in ignored_images.keys()])} images due to errors - see the file for more details.")
   


if __name__ == "__main__":
    main()
    
    
