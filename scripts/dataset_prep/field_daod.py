import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils import coco

def remove_categories(src_coco):
    src_images = src_coco["images"]
    src_anns = src_coco["annotations"]
    total_images = len(src_images)
    images_to_remove = set()
    cat_id_to_name = coco.category_id_to_name(src_coco["categories"])

    for i,ann in enumerate(src_anns):
        clss = cat_id_to_name[ann["category_id"]]
        if clss not in classes_to_keep:
            images_to_remove.add(ann["image_id"])
    
    anns = coco.filter_annotations_out(all_annotations=src_anns, image_ids_to_filter=images_to_remove)
    images = coco.filter_images_out(all_images=src_images, image_ids_to_filter=images_to_remove)

    src_coco["images"] = images
    src_coco["annotations"] = anns

def reindex_categories(src_coco):
    old_to_new_id = {}
    
    src_cat_id_to_name = coco.category_id_to_name(src_coco["categories"])
    new_cat_name_to_id = coco.category_name_to_id(new_classes)
    
    for ann in src_coco["annotations"]:
        name = src_cat_id_to_name[ann["category_id"]]
        new_id = new_cat_name_to_id[name]
        ann["category_id"] = new_id
    
    src_coco["categories"] = new_classes

classes_to_keep = ["aranae","carabidae","diptera-hymenoptera","isopoda","myriapoda","staphylinidae"]
new_classes = []
for i,clss in enumerate(classes_to_keep):
    new_classes.append({"id": i, "name": clss})

src_dir = "annotations/osr-fields/"
lg_coco_file = src_dir + "lg/LG_OSR.json"
gh_coco_file = src_dir + "gh/GH_OSR.json"
lg_coco = coco.load_from_file(lg_coco_file)
gh_coco = coco.load_from_file(gh_coco_file)

# Remove the unneeded categories
remove_categories(lg_coco)
remove_categories(gh_coco)

# Reindex remaining categories
reindex_categories(lg_coco)
reindex_categories(gh_coco)

coco.save_to_file(json_obj=lg_coco, path=lg_coco_file)
coco.save_to_file(json_obj=gh_coco, path=gh_coco_file)

