#!/bin/bash

model_path=$1

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

python src/test_single.py \
    -root_dir ${base_dir}/data_model/ \
    -split test \
    -batch_size 1 \
    -model_path ${model_path}