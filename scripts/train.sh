#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python ../train.py \
    --model_name "dl4mt" \
    --reload \
    --config_path "../configs/dl4mt_config.yaml" \
    --log_path "./log" \
    --saveto "./save/" \
    --use_gpu