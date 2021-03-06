#!/bin/bash

dataset=$1
split=$2
model_path=$3

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

python src/evaluate.py \
    -root_dir ${base_dir}/data_model/ \
    -dataset ${dataset} \
    -split ${split} \
    -batch_size 2 \
    -model_path ${model_path}
