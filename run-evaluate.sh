#!/bin/bash

dataset=$1
model_path=$2

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

python src/evaluate.py \
    -root_dir "${base_dir}/data_model/" \
    -dataset ${dataset} \
    -split test \
    -batch_size 2 \
    -model_path ${model_path} \
    -word_embeddings "${base_dir}/data_model/${dataset}/embedding"

#CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate.py \
#    -root_dir "${work_dir}/data_model/" \
#    -dataset ${dataset} \
#    -split dev \
#    -model_path "$data_dir/run.*/m_*.pt"

#model_path=$(head -n1 ${data_dir}/dev_best.txt)

#CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate.py
#    -root_dir "${work_dir}/data_model/"
#    -dataset ${dataset}
#    -split test
#    -model_path ${model_path}