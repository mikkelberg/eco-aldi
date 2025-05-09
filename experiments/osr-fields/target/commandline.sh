#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
ln -sfT /mnt/data0/martez/ datasets

# Run experiment
python ../../../aldi/tools/train_net.py --config input/GH-OSR_RCNN-FPN_oracle.yaml