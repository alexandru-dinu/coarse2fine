#!/bin/bash

dataset=$1
model_path=$2

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

python src/test-single.py \
    -root_dir "${base_dir}/data_model/" \
    -dataset ${dataset} \
    -model_path ${model_path} \
    -use_custom_embeddings \
    -word_embeddings "${base_dir}/data_model/glove-fine-tuned-5000" \
    -word_emb_size 300

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