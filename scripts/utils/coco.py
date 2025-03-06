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