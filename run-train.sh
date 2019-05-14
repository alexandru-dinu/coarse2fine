#!/bin/bash

dataset=$1
base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}
emb_dir=${base_dir}/../embeddings

emb_exp_dir=${emb_dir}/2019-05-14_19-16-59-python-so-50

python src/train.py \
    -root_dir ${base_dir}/data_model/ \
    -dataset ${dataset} \
    -rnn_size 300 \
    -decoder_input_size 200 \
    -layers 1 \
    -learning_rate 0.002 \
    -epochs 50 \
    -global_attention "dot" \
    -attn_hidden 0 \
    -dropout 0.3 \
    -dropout_i 0.3 \
    -lock_dropout \
    -copy_prb hidden \
    -batch_report_every 25 \
    -start_checkpoint_at 10 \
    -word_emb_size 50 \
    -use_custom_embeddings \
    -vocab_file ${emb_exp_dir}/*.vocab \
    -pt_embeddings ${emb_dir}/glove.6B.50d.txt \
    -word_embeddings ${emb_exp_dir}/*.emb \
    -pt_factor 0.1 \
    -ft_factor 0.9 \
    -seed 1234 \
    -cuda \
    -exp_name django-ft-emb-6B.50d \
