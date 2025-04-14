#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
# Link name MUST be 'datasets' to be compatible with ALDI
ln -sfT /mnt/data0/martez/ datasets

# Run experiment
export CUDA_VISIBLE_DEVICES=2 # run on the new freja GPU
python ../../../aldi/tools/train_net.py --config input/aldi++.yaml #--num-gpus 1
# python ../../../aldi/tools/train_net.py --config input/concon_RCNN-FPN_source-only.yaml --resume

# Evaluate on test-data:
# python ../../../aldi/tools/train_net.py --eval-only --num-gpus 4 --config-file input/pitfall-cameras_RCNN-FPN_oracle.yaml MODEL.WEIGHTS output/training/pitfall-cameras_val_model_best.pth DATASETS.TEST '("pitfall-cameras_test",)'