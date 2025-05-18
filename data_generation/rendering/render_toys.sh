#!/bin/bash

data_path='/data/odme/toys4k_blend_files'
# output_path='/data/datasets/LSME/toys_rendering_output_new_thesis_2unknowns'
# output_path='/data/datasets/LSME/toys_rendering_output_new_thesis_1obj'
output_path='/data/datasets/LSME/toys_rendering_output_new_thesis_viz'


blender_path='/home/ant/Downloads/blender-3.6.20-linux-x64/blender'
# scene_config_path='/home/ant/develop/LSME/data_generation/common/toys_scene_configs_me_easy'
scene_config_path='/home/ant/develop/LSME/data_generation/common/toys_scene_configs_2unknown_test'

# scene_config_path='/home/ant/develop/LSME/data_generation/common/toys_scene_configs_l3_test'

overwrite=False

python wrapper.py \
    --start=$1 \
    --end=$2 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --overwrite=$overwrite \
    --dataset_type=toys 2>&1 | tee datagen_log_modelnet.txt
