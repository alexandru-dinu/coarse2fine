#!/bin/bash

dataset=$1
base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

model_path=${data_dir}/pretrain.pt

python evaluate.py \
    -root_dir "${base_dir}/data_model/" \
    -dataset ${dataset} \
    -split test \
    -model_path ${model_path}
