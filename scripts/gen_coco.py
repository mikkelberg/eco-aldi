import json
import pandas as pd
import os
import re
import argparse
from PIL import Image

from utils.utils import load_json_from_file, save_json_to_file, read_image_size
import utils.pitfall_cameras_utils as pc
import dataset_prep.extract_coco_categories as cats

def image(id, file_name, images_dir):
    width, height = read_image_size(images_dir+file_name)

    image = {}
    image["id"] = id
    image["height"] = height 
    image["width"] = width 
    image["file_name"] = file_name
    return image

annotation_counter = 0
def annotation(image_id, category_id, bbox):
    global annotation_counter
    annotation = {}
    annotation["id"] = annotation_counter
    annotation["image_id"] = image_id
    annotation["category_id"] = category_id
    annotation["segmentation"] = []
    annotation["area"] = bbox[3]*bbox[3]
    annotation["bbox"] = [bbox[0], bbox[1], bbox[2], bbox[3]]
    annotation["iscrowd"] = 0

    annotation_counter += 1 # incr the global counter
    return annotation

def gen_coco(info, images, annotations, categories):
    coco = {}
    coco["info"] = info
    coco["license"] = None
    coco["images"] = images
    coco["annotations"] = annotations
    coco["categories"] = categories
    return coco

def add_to_ignored_images(img_folder_name, explanation, img_id, og_ann):
    og_filename = img_id.split("_")[-1]
    if img_folder_name not in ignored_images.keys():
        ignored_images[img_folder_name] = {} 
    if img_id not in ignored_images[img_folder_name].keys():
        ignored_images[img_folder_name][img_id] = {"explanation": explanation, "original_filename": og_filename, "orignal_csv_file": og_ann}
    
                
def check_and_clean_category_name(category_name, categories_to_remove, category_to_group):
    category_name = cats.normalise_category_name(category_name)
    if category_name in categories_to_remove.keys():
        return category_name, True
    
    if category_name in category_to_group.keys(): 
        category_name = category_to_group[category_name]

    return category_name, False

def pitfall_cameras_coco(src_dir, dest_dir, categories, images_dir):
    """Runs through a given directory of vgg-csv annotation files and converts all of them to COCO JSON format."""
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".csv")))]
    total_files = len(files)
    cam_and_date_to_img_folder_name = pc.gen_camdate_to_imgfolder_dict()

    categoriesss= load_json_from_file("annotations/categories.json")["categories"]
    category_to_group = {}
    for cat in categoriesss.keys():
        for contained_cat in categoriesss[cat]["contains"]:
            category_to_group[contained_cat] = cat
    categories_to_remove = load_json_from_file("annotations/categories.json")["remove"]
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

    # loop through all source csv-files
    for index, filename in enumerate(files, start=1): 
        src_file_path = os.path.join(src_dir, filename)
        data = pc.get_dataframe_from_anncsv(csv_file_path=src_file_path)

        date = pc.get_datetime_from_anncsv_filename(filename)
        cam = pc.get_camera_from_anncsv_filename(filename)
        field = pc.get_field_from_anncsv_filename(filename)
        flash = "_fl" if filename.split(" ")[-1].startswith("on") else ""
        img_folder_name = cam_and_date_to_img_folder_name[f"{cam}-{date}"]
        
        # Create info entry
        info = pc.gen_coco_info(img_folder_name=img_folder_name)

        # Create images entries
        images = []
        imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        for row in imagedf.itertuples():
            images.append(image(id=row.fileid, file_name=row.filename, images_dir=images_dir))

        # Create annotations entries
        images_to_clean_out = set()
        annotations = []
        for row in data.itertuples():
            if row.region_count <= 0:
                continue
            # if there is a detection, append the annotation
            category_name = pc.extract_category_name_from_region_attributes(row.region_attributes)

            no_insect_label_but_was_annotated = not bool(category_name)
            if no_insect_label_but_was_annotated: 
                explanation = "Incomplete annotation: No insect class in annotation (region_attributes)."
                add_to_ignored_images(img_folder_name=img_folder_name, explanation=explanation,img_id=row.fileid, og_ann=filename)
                images_to_clean_out.add(row.fileid)
                continue

            category_name, remove = check_and_clean_category_name(category_name, categories_to_remove, category_to_group)
            if remove:
                explanation = f"Invalid category: {category_name}, {categories_to_remove[category_name]}."
                add_to_ignored_images(img_folder_name=img_folder_name, explanation=explanation,img_id=row.fileid, og_ann=filename)
                images_to_clean_out.add(row.fileid)
                continue

            category_id = category_name_to_id[category_name]
            shape = json.loads(row.region_shape_attributes)
            ann = annotation(image_id=row.fileid, category_id=category_id, bbox=[shape["x"], shape["y"], shape["width"], shape["height"]])
            annotations.append(ann)

        coco = gen_coco(info, images, annotations, categories)
        save_json_to_file(json_obj=coco, path=dest_dir+img_folder_name+date+flash+".json")
        # Print progress every 5 files
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")
    
    print(f"Converted all the csv-annotations in {src_dir}")

def image_entries_from_dir(images_dir):
    images = []
    stop = 0
    img_id = 1
    for root, _, files in os.walk(images_dir):
        print(f"Processing folder: {root}.")
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue  # Ensure it's an image file
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(image_path, images_dir)
            images.append(image(id=img_id, file_name=relative_path, images_dir=images_dir))
            img_id += 1
        stop += 1
        if stop == 5: break
    return images

    
def annotations_for_concon(via_file, df):
    img_metadata = load_json_from_file(via_file)["_via_img_metadata"]
    return

def concon_coco(meta_csv, dest_path, categories, images_dir):
    df = pd.read_csv(meta_csv, dtype={"date": str})
    df["date"].astype(str)
    for row in df.itertuples():
        img_folder_name = row.date + "/" + row.camera + "/"
        #print(img_folder_name)
    print(df)
    
    info = "Annotations for object detections in the images collection in the ECOSTACK-project's experiment with controlled conditions, where various insects where photographed in boxes with paper or soil background (we use only the paper images here). The images were taken in June-August 2023. Converted to COCO-format by Stinna Danger and Mikkel Berg for their thesis project at Aarhus University at the Department of Computer Science with biodiversity group at the Department of Ecoscience."
    
    images = image_entries_from_dir(images_dir)

    via_file = "/annotations/controlled-conditions/src-files/040523 CAM8.json"
    annotations = annotations_for_concon(via_file, df)

    coco = gen_coco(info, images, annotations, categories)
    save_json_to_file(json_obj=coco, path=dest_path)
    print(f"Generated coco-annotations.")


ignored_images = {} # {"csv_file_name": [{"filename": "bla", "explanation": "bla", "original_filename": "bla", "original_csv_file" : "bla"}, ...], ...}
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert each VGG-CSV annotations in a directory to COCO JSON format.")
    parser.add_argument("dataset_name", help="E.g. pitfall-cameras (matches the directory)")
    parser.add_argument("images_dir", help="Path to images directory.")

    # python scripts/gen_coco.py pitfall-cameras ../../../mnt/data0/martez/pitfall-cameras/images/

    # Parse the arguments
    args = parser.parse_args()

    # Generate categories-entry
    
    root = "annotations/"

    categories_file = root + "/categories.json"
    categories_info = load_json_from_file(categories_file)
    categories = [{"id": categories_info["categories"][cat]["id"], "name": cat} for cat in categories_info["categories"].keys()]
    print(categories)

    root = "annotations/"
    if args.dataset_name == "pitfall-cameras":
        root = root + args.dataset_name
        src_dir = root + "/originals/"
        dest_dir = root + "/originals-converted/"
        ignored_images_path = root + "/info/ignored_images.json"

        # Convert
        pitfall_cameras_coco(src_dir=src_dir, dest_dir=dest_dir, categories=categories, images_dir=args.images_dir)
    elif args.dataset_name == "controlled-conditions":
        root = root + args.dataset_name
        meta_csv = root + "/info/meta.csv"
        dest_path = root + "/controlled-conditions_all.json"
        ignored_images_path = root + "/info/ignored_images.json"

        concon_coco(meta_csv=meta_csv, dest_path=dest_path, categories=categories, images_dir=args.images_dir)

    # Store record of ignored images
    with open(ignored_images_path, 'w') as f:
        json.dump(ignored_images, f, indent=4)

    print(f"Ignored {sum([len(ignored_images[key]) for key in ignored_images.keys()])} images due to errors - see the file for more details.")

if __name__ == "__main__":
    main()