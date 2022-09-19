#! /bin/bash

set -e

mod=${1}
device=1
echo "mod: ${mod}"

cd src

# train recursion
if [[ ${mod} == "train_math23k" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_recursion_math23k.py \
        --dataset_name 'math23k' \
        --log_text '(v3) 不同的输出词表' \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --fold -1 \
        --save_model \
        --op_seq_mode v3

fi

if [[ ${mod} == "train_gsm8k" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_recursion_gsm8k.py \
        --dataset_name 'gsm8k' \
        --log_text '(v1)' \
        --data_path '../data/GSM8k' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"model_name":"bert-base-uncased"}' \
        --op_seq_mode v1 \

fi

# debug
if [[ ${mod} == "debug_math23k" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_recursion_math23k.py \
        --dataset_name 'math23k' \
        --log_text '(debug)' \
        --data_path '../data/Math23K' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":30}' \
        --head 1000 \
        --fold -1 \
        --op_seq_mode v2
fi

if [[ ${mod} == "debug_dag" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_recursion_dag.py \
        --dataset_name 'dag' \
        --log_text '(debug)' \
        --data_path '../data/DAG' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":21}' \
        --head 1000 \
        --op_seq_mode v2
fi

if [[ ${mod} == "debug_gsm8k" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_recursion_gsm8k.py \
        --dataset_name 'gsm8k' \
        --log_text '(debug)' \
        --data_path '../data/GSM8k' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":30}' \
        --head 200 \
        --op_seq_mode v1 \
        --debug
fi

cd ..

# backup

# # train deduce
# if [[ ${mod} == "train_deduce" ]];
# then
#     CUDA_VISIBLE_DEVICES=${device} python train_deduce.py \
#         --data_path '../data/Math23K' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --fold -1
# fi

# # train seq2seq
# if [[ ${mod} == "train_seq2seq" ]];
# then
#     CUDA_VISIBLE_DEVICES=${device} python train_seq2seq.py \
#         --log_text 'baseline model (BERT2seq)' \
#         --data_path '../data/Math23K' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --fold -1
# fi
