#!/bin/bash

dataset=$1
base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}


python src/train.py \
    -root_dir "${base_dir}/data_model/" \
    -dataset ${dataset} \
    -rnn_size 300 \
    -word_vec_size 250 \
    -decoder_input_size 200 \
    -layers 1 \
    -start_checkpoint_at 15 \
    -learning_rate 0.002 \
    -epochs 25 \
    -global_attention "dot" \
    -attn_hidden 0 \
    -dropout 0.3 \
    -dropout_i 0.3 \
    -lock_dropout \
    -copy_prb hidden \
    -exp django-0

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
