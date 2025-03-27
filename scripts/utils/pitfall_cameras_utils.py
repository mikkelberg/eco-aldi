import re
import json
from PIL import Image
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pitfall_cameras_utils as pc
import utils.utils as utils 

INFO_FILE_PATH =  "annotations/pitfall-cameras/info/info.json"
IGNORED_IMAGES_PATH =  "annotations/pitfall-cameras/info/ignored_images.json"
IMAGES_FOLDER = "../ERDA/bugmaster/datasets/pitfall-cameras/images/"

LOCATIONS = ['GH_OSR_HF2G', 'GH_OSR_LF1F', 'GH_OSR_LF2E', 'GH_OSR_NARS26', 'GH_OSR_NARS30', 'LG_OSR_HF2F', 'LG_OSR_LF1D', 'LG_OSR_LF1G', 'LG_OSR_LF2F', 'LG_OSR_LS3E', 'LG_WWH_NARS30', 'WW_OSR_HF2F', 'WW_OSR_LF1D', 'WW_OSR_LF1G', 'WW_OSR_LF2F', 'WW_OSR_LS3E']

def get_original_anncsv_filename(date:str, camera, flash=False):
    date_list = date.split("-")
    date_list.reverse()
    src_date = "".join(date_list)
    flash = "on" if flash else "off"
    return f"{src_date} {camera} Flash {flash}_csv.csv"

def get_camera_from_anncsv_filename(csv_name):
    return csv_name.split(" ")[1]

def get_datetime_from_anncsv_filename(csv_name):
    og_date = csv_name.split(" ")[0]
    date = re.findall('..', og_date)
    date.reverse()
    date = "-".join(date)
    return date

def get_field_from_anncsv_filename(csv_name):
    date = get_datetime_from_anncsv_filename(csv_name)
    cam = get_camera_from_anncsv_filename(csv_name)
    img_folder_name = gen_camdate_to_imgfolder_dict()[f"{cam}-{date}"]
    return get_specs_from_info(img_folder_name)["field"]
    
def gen_camdate_to_imgfolder_dict():
    with open(INFO_FILE_PATH, "r") as f:
        folders = json.load(f)["folders"]
    date_and_camera_to_folder_name = {}
    for folder in folders.keys():
        camera = folders[folder]["specs"]["camera"]
        dates = folders[folder]["specs"]["dates"]
        for date in dates:
            date_and_camera_to_folder_name[f"{camera}-{date}"] = folder
    return date_and_camera_to_folder_name

def gen_coco_info(img_folder_name):
    info_json = utils.load_json_from_file(INFO_FILE_PATH)
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

def create_coco_filename(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}.json"

def get_specs_from_info(img_folder_name, info_file_path=INFO_FILE_PATH):
    info = {}
    with open(info_file_path, 'r') as f:
        info = json.load(f)
        specs = info["folders"][img_folder_name]["specs"]
    return specs

def get_img_folder_name_from_specs(field, crop, camera):
    return f"{field}-{crop}-{camera}"

def get_file_prefix_from_specs(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}_"

def get_location_prefix_from_image_filename(filename: str):
    """ takes FIELD_CROP_CAMERA_DATE_IMGNAME.JPG
        returns FIELD_CROP_CAMERA """
    return "_".join(filename.split("_")[0:3]) 
def get_img_folder_name_from_image_filename(filename: str):
    """ takes FIELD_CROP_CAMERA_DATE_IMGNAME.JPG
        returns FIELD-CROP-CAMERA """
    return "-".join(filename.split("_")[0:3]) 

def get_dataframe_from_anncsv(csv_file_path):
    filename = csv_file_path.split("/")[-1]
    
    # Get name of folder containing these images
    date = get_datetime_from_anncsv_filename(filename)
    cam = get_camera_from_anncsv_filename(filename)
    cam_and_date_to_img_folder_name = gen_camdate_to_imgfolder_dict()
    img_folder_name = cam_and_date_to_img_folder_name[f"{cam}-{date}"]
    
    # Get the prefix prepended to the img file names
    specs = get_specs_from_info(img_folder_name)
    flash = filename.split(" ")[-1].startswith("on")
    file_prefix = get_file_prefix_from_specs(field=specs["field"], crop=specs["crop"], camera=specs["camera"], date=date, flash=flash)
    
    # read file
    data = pd.read_csv(csv_file_path, on_bad_lines='skip')

    data['fileid'] = file_prefix + data['filename'].astype(str) # create fileid column
                                                                # id is the file prefix prepended to the original filename (this corresponds to the correct filename in OUR system)
    data['filename'] = img_folder_name + "/" + data['fileid'].astype(str) # update the filename to have the entire path (and of course the correct filename, which we just saved in the fileid-column)
    
    return data

def extract_category_name_from_region_attributes(attr):
    try:
        attr_dict = json.loads(attr)
        #keys = list(attr_dict.keys())
        #if ("Insect ID" not in keys) and ("Insect" not in keys) and ("Insects" not in keys): print(attr_dict)
        for key in ["Insect ID"]:
            # retrieve label from this format: {"Insect": "Carabid"}
            if key in attr_dict:
                return attr_dict[key]
        for key in ["Insect", "Insects"]:
            # retrieve label from this format: {"Insect": {"Carabid": True}}
            if key in attr_dict:
                keys = list(attr_dict[key].keys())
                if len(keys)==0:
                    return None # no label
                return keys[0] 
    except json.JSONDecodeError:
        return None # ignore malformed json
    return None 

def ignored_img(filename, explanation, og_csv_name):

    return {"filename": filename, "explanation": explanation, "original_filename": og_filename, "orignal_csv_file": og_csv_name}

name_mappings = { # corrections for typos and redundancies that weren't caught by stanadardizing the name
    "arachnid": "arachnida",
    "amara sp": "amara",
    "braconid": "braconidae",
    "carabid": "carabidae", 
    "carabid unknown": "carabidae",  
    "carabids": "carabidae",  
    "chalcidae": "chalcididae",
    "dipteran larvae": "diptera larvae",
    "gnaposidae": "gnaphosidae",
    "isopod": "isopoda",
    "linyphiiidae": "linyphiidae",
    "molllusca": "mollusca",
    "molllusc": "mollusca",
    "mollusc": "mollusca",
    "molluska": "mollusca",
    "myriapod": "myriapoda",
    "phyllotreta sp": "phyllotreta",
    "poecilius cupreus": "poecilus cupreus",
    "psyllidoes chrysocephalus": "psylliodes chrysocephalus",
    "spider": "arachnida",
    "tachyporus hyphorum": "tachyporus hypnorum",
    "tachyprous hypnorum": "tachyporus hypnorum",
    "unsure": "unknown",
    }

names_to_group = {
    "tachyporus hypnorum": "staphylinidae",
    "pterostichus melanarius": "carabidae",
    "pterostichus madidus": "carabidae",
    "psylliodes chrysocephalus": "chrysomelidae",
    "poecilus cupreus": "carabidae",
    "phyllotreta": "chrysomelidae",
    "opiliones": "arachnida",
    "notiophilus biguttatus": "carabidae",
    "nebria brevicollis": "carabidae",
    "lycosidae": "arachnida",
    "linyphiidae": "arachnida",
    "harpalus rufipes": "carabidae",
    "harpalus affinis": "carabidae",
    "gnaphosidae": "arachnida",
    "coccinella": "coccinellidae",
    "cantharis rustica": "cantharidae",
    "cantharis rufa": "cantharidae",
    "cantharis lateralis": "cantharidae",
    "brassicogethes aeneus": "nitidulidae",
    "brassicogethes": "nitidulidae",
    "anchomenus dorsalis": "carabidae",
    "amara": "carabidae"
}

categories_to_set_to_unknown = ["coleoptera", "parasitoid"]