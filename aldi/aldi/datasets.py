from detectron2.data.datasets import register_coco_instances

# Controlled-conditions
register_coco_instances("controlled-conditions_train",   {},         "datasets/controlled-conditions/dataset/annotations/controlled-conditions_train.json",                    "datasets/controlled-conditions/dataset/images/")
register_coco_instances("controlled-conditions_val",   {},         "datasets/controlled-conditions/dataset/annotations/controlled-conditions_val.json",                    "datasets/controlled-conditions/dataset/images/")
register_coco_instances("controlled-conditions_test",   {},         "datasets/controlled-conditions/dataset/annotations/controlled-conditions_test.json",                    "datasets/controlled-conditions/dataset/images/")

register_coco_instances("controlled-conditions_train-downsample2000",   {},         "datasets/controlled-conditions/dataset/annotations/controlled-conditions_train-downsample2000.json",                    "datasets/controlled-conditions/dataset/images/")

# Pitfall-cameras
register_coco_instances("pitfall-cameras_train",   {},         "datasets/pitfall-cameras/annotations/pitfall-cameras_train.json",                    "datasets/pitfall-cameras/images/")
register_coco_instances("pitfall-cameras_val",   {},         "datasets/pitfall-cameras/annotations/pitfall-cameras_val.json",                    "datasets/pitfall-cameras/images/")
register_coco_instances("pitfall-cameras_test",   {},         "datasets/pitfall-cameras/annotations/pitfall-cameras_test.json",                    "datasets/pitfall-cameras/images/")

# Cityscapes 
register_coco_instances("cityscapes_train", {},         "datasets/cityscapes/annotations/cityscapes_train_instances.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances.json",                    "datasets/cityscapes/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {},   "datasets/cityscapes/annotations/cityscapes_train_instances_foggyALL.json",   "datasets/cityscapes/leftImg8bit_foggy/train/")
register_coco_instances("cityscapes_foggy_val", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")
# for evaluating COCO-pretrained models: category IDs are remapped to match
register_coco_instances("cityscapes_foggy_val_coco_ids", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL_coco.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")

# Sim10k
register_coco_instances("sim10k_cars_train", {},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images/")
register_coco_instances("cityscapes_cars_train", {},         "datasets/cityscapes/annotations/cityscapes_train_instances_cars.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_cars_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances_cars.json",                    "datasets/cityscapes/leftImg8bit/val/")

# CFC
register_coco_instances("cfc_train", {},         "datasets/cfc_daod/coco_labels/cfc_train.json",                  "datasets/cfc_daod/images/cfc_train/")
register_coco_instances("cfc_val",   {},         "datasets/cfc_daod/coco_labels/cfc_val.json",                    "datasets/cfc_daod/images/cfc_val/")
register_coco_instances("cfc_channel_train", {},         "datasets/cfc_daod/coco_labels/cfc_channel_train.json",                  "datasets/cfc_daod/images/cfc_channel_train/")
register_coco_instances("cfc_channel_test",   {},         "datasets/cfc_daod/coco_labels/cfc_channel_test.json",                    "datasets/cfc_daod/images/cfc_channel_test/")