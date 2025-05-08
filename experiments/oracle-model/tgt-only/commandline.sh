#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
# Link name MUST be 'datasets' to be compatible with ALDI
ln -sfT /mnt/data0/martez/ datasets

# Run experiment
python ../../../aldi/tools/train_net.py --config input/oracle_tgt-only.yaml #--num-gpus 2

# Evaluate on test-data:
# python ../../../aldi/tools/train_net.py --eval-only --num-gpus 4 --config-file input/pitfall-cameras_RCNN-FPN_oracle.yaml MODEL.WEIGHTS output/training/pitfall-cameras_val_model_best.pth DATASETS.TEST '("pitfall-cameras_test",)'