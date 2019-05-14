#!/bin/bash

dataset=$1
model_path=$2

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}
emb_dir=${base_dir}/../embeddings

python src/evaluate.py \
    -root_dir ${base_dir}/data_model/ \
    -dataset ${dataset} \
    -split test \
    -batch_size 2 \
    -model_path ${model_path}

#    -word_emb_size 50 \
#    -use_custom_embeddings \
#    -word_embeddings ${emb_dir}/2019-05-14_19-16-59-python-so-50/python-so.emb \
#    -pt_embeddings ${emb_dir}/glove.6B.50d.txt