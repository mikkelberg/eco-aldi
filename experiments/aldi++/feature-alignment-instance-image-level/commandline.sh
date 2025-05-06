#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
# Link name MUST be 'datasets' to be compatible with ALDI
ln -sfT /mnt/data0/martez/ datasets

# Run experiment
export CUDA_VISIBLE_DEVICES=1 # run on target GPU
python ../../../aldi/tools/train_net.py --config input/aldi++.yaml

# Evaluate on test-data:
#python ../../../aldi/tools/train_net.py --eval-only --num-gpus 3 --config-file input/aldi++.yaml MODEL.WEIGHTS input/controlled-conditions_val_model_best.pth