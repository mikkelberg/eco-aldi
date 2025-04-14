#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
# Link name MUST be 'datasets' to be compatible with ALDI
ln -sfT /mnt/data0/martez/ datasets

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_SUBSYS=GRAPH
export TORCH_NCCL_TRACE_BUFFER_SIZE=20
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1


# Run experiment
export CUDA_VISIBLE_DEVICES=2 # run on the new freja GPU
python ../../../aldi/tools/train_net.py --config input/aldi++.yaml --num-gpus 1

# Evaluate on test-data:
# python ../../../aldi/tools/train_net.py --eval-only --num-gpus 4 --config-file input/pitfall-cameras_RCNN-FPN_oracle.yaml MODEL.WEIGHTS output/training/pitfall-cameras_val_model_best.pth DATASETS.TEST '("pitfall-cameras_test",)'