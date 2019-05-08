#!/bin/bash

dataset=$1
base_dir=$(dirname "$(readlink -f $0)")
data_dir=${base_dir}/data_model/${dataset}


python src/preprocess.py \
   -root_dir "${base_dir}/data_model/" \
   -dataset ${dataset} \
   -src_words_min_frequency 3 -tgt_words_min_frequency 5