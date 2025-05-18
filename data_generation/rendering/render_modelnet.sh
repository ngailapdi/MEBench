#!/bin/bash

data_path='/data/datasets/ShapeNetCore.v2'
output_path='/home/ant/develop/LSME/data_generation/shapenet_test2'
blender_path='/home/ant/Downloads/blender-3.6.20-linux-x64/blender'
scene_config_path='/home/ant/develop/LSME/data_generation/common/shapenet_scene_configs_test'

CUDA_VISIBLE_DEVICES=1
python wrapper.py \
    --start=0 \
    --end=1 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=shapenet 2>&1 | tee datagen_log_modelnet.txt
