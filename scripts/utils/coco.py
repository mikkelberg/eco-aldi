def filter_annotations(all_annotations, image_ids_to_filter):
    return [ann for ann in all_annotations if ann["image_id"] in image_ids_to_filter]

def filter_images(all_images, image_ids_to_filter):
    return [img for img in all_images if img["id"] in image_ids_to_filter]

def filter_annotations_out(all_annotations, image_ids_to_filter):
    return [ann for ann in all_annotations if ann["image_id"] not in image_ids_to_filter]

def filter_images_out(all_images, image_ids_to_filter):
    return [img for img in all_images if img["id"] not in image_ids_to_filter]
