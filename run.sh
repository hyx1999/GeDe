#! /bin/bash

set -e

mod=${1}
device=2
echo "mod: ${mod}"

cd src

# train deduce
if [[ ${mod} == "train_recursion" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_recursion.py \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --fold -1 \
        --save_model
fi

# train deduce
if [[ ${mod} == "train_deduce" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_deduce.py \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --fold -1
fi

# train seq2seq
if [[ ${mod} == "train_seq2seq" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_seq2seq.py \
        --log_text 'baseline model (BERT2seq)' \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --fold -1
fi

# debug
if [[ ${mod} == "debug" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_recursion.py \
        --log_text '(debug)' \
        --data_path '../data/Math23K' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":100}' \
        --head 1000 \
        --fold -1
fi

cd ..
