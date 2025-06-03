#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
ln -sfT /mnt/data0/martez/ datasets

# Run experiment
export CUDA_VISIBLE_DEVICES=0
python ../../../aldi/tools/train_net.py --config input/aldi++.yaml