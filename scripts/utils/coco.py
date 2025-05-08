import cv2
import json

def save_to_file(json_obj, path):
    with open(path, "w") as f:
        json.dump(json_obj, f, indent=4)

def load_from_file(path):
    with open(path, "r") as f:
        json_obj = json.load(f)
    return json_obj

def filter_annotations(all_annotations, image_ids_to_filter):
    return [ann for ann in all_annotations if ann["image_id"] in image_ids_to_filter]

def filter_images(all_images, image_ids_to_filter):
    return [img for img in all_images if img["id"] in image_ids_to_filter]

def filter_annotations_out(all_annotations, image_ids_to_filter):
    return [ann for ann in all_annotations if ann["image_id"] not in image_ids_to_filter]

def filter_images_out(all_images, image_ids_to_filter):
    return [img for img in all_images if img["id"] not in image_ids_to_filter]

def get_positive_and_negative_samples(images, annotations):
    positive_sample_ids = {ann["image_id"] for ann in annotations}
    positive_samples = []
    negative_samples = []
    for img in images:
        if img["id"] in positive_sample_ids:
            positive_samples.append(img)
        else:
            negative_samples.append(img)
    return positive_samples, negative_samples

def get_positive_and_negative_sample_ids_lists(images, annotations):
    positive_sample_ids = {ann["image_id"] for ann in annotations}
    negative_sample_ids = {img["id"] for img in images if img["id"] not in positive_sample_ids}
    return list(positive_sample_ids), list(negative_sample_ids)

def load_image_to_bbox_and_cat_pairs_from_annotations(coco_json: dict):
    image_to_bbox_and_cat_pairs = {}
    # initialise all with no annotations,  []
    for img in coco_json["images"]:
        image_to_bbox_and_cat_pairs[img["id"]] = []
    # add the annotations, [(bbox, cat_id), (bbox, cat_id), ...]
    for ann in coco_json["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        cat_id = ann["category_id"]
        image_to_bbox_and_cat_pairs[image_id].append((bbox, cat_id))
    return image_to_bbox_and_cat_pairs


def image_id_to_filename(coco_images):
    return {img["id"]: img["file_name"] for img in coco_images}

def category_id_to_name(coco_categories):
    return {cat["id"]: cat["name"] for cat in coco_categories}

def category_name_to_id(coco_categories):
    return {cat["name"]: cat["id"] for cat in coco_categories}

# Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

# Load image by its ID
def load_image(image_id, coco_images, image_dir):
    img_info = next(img for img in coco_images if img['id'] == image_id)
    img_path = f"{image_dir}/{img_info['file_name']}"
    image = cv2.imread(img_path)
    return image, img_info