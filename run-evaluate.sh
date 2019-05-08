#!/bin/bash

dataset=$1
model_path=$2

base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}

python src/evaluate.py \
    -root_dir ${base_dir}/data_model/ \
    -dataset ${dataset} \
    -split test \
    -batch_size 2 \
    -model_path ${model_path} \
    -vocab_file ${base_dir}/data_model/comp-sci-corpus-thr20000-window10-tfidf.vocab