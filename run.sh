#! /bin/bash

set -e

mod=${1}
log=${2}
device=3
echo "mod: ${mod}"

cd src

if [[ ${mod} == "train_math23k" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
        --dataset_name 'math23k' \
        --log_text ${log} \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --fold -1 \
        --save_model \
        --expr_mode v3

fi

if [[ ${mod} == "train_svamp" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --dataset_name 'svamp' \
        --log_text ${log} \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --expr_mode v3

fi

# debug
if [[ ${mod} == "debug_math23k" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
        --dataset_name 'math23k' \
        --log_text '(debug)' \
        --data_path '../data/Math23K' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":30}' \
        --head 1000 \
        --fold -1 \
        --expr_mode v3 \
        --debug
fi

if [[ ${mod} == "debug_svamp" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --dataset_name 'svamp' \
        --log_text '(debug)' \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"num_epochs":30}' \
        --head 1000 \
        --expr_mode v3 \
        --debug

fi

if [[ ${mod} == "debug_mathtoy" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_mathtoy.py \
        --dataset_name 'mathtoy' \
        --log_text '(debug)' \
        --data_path '../data/MathToy' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"num_epochs":30}' \
        --head 1000 \
        --expr_mode v2 \
        --debug
fi

if [[ ${mod} == "debug_webqsp" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_webqsp.py \
        --dataset_name 'webqsp' \
        --log_text '(debug)' \
        --data_path '../data/WebQSP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
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
