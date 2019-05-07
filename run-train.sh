#!/bin/bash

dataset=$1
base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}


python src/train.py \
    -root_dir "${base_dir}/data_model/" \
    -dataset ${dataset} \
    -rnn_size 300 \
    -word_emb_size 250 \
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
    -batch_report_every 25 \
    -start_checkpoint_at 5 \
    -word_emb_size 300 \
    -exp_name django-ft-embs-5000 \
    -use_custom_embeddings -word_embeddings ../data_model/glove-fine-tuned-1000 \
    -word_emb_size 300 \
    -cuda
