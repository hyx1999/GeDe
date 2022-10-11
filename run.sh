#! /bin/bash

set -e

mod=${1}
log='log:'${2}
device=0
echo "mod: ${mod}"

cd src

if [[ ${mod} == "train_svamp" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --model_type 'test' \
        --dataset_name 'svamp' \
        --log_text ${log} \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","bert_lr":2e-5,"max_step_size":5,"save_result":true,"num_epochs":1000,"batch_size":8}'
fi


if [[ ${mod} == "debug_svamp" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --model_type 'test' \
        --dataset_name 'svamp' \
        --log_text '(debug)' \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"roberta-base","bert_lr":2e-5,"max_step_size":5,"save_result":true}' \
        --head 1000 \
        --debug

fi

cd ..

# backup

# if [[ ${mod} == "train_math23k" ]];
# then

#     CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
#         --dataset_name 'math23k' \
#         --log_text ${log} \
#         --data_path '../data/Math23K' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --fold -1 \
#         --save_model \
#         --expr_mode v3

# fi

# # debug
# if [[ ${mod} == "debug_math23k" ]];
# then
#     CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
#         --dataset_name 'math23k' \
#         --log_text '(debug)' \
#         --data_path '../data/Math23K' \
#         --load_model_dir 'models_test' \
#         --save_model_dir 'models_test' \
#         --cfg '{"num_epochs":30}' \
#         --head 1000 \
#         --expr_mode v3 \
#         --debug
# fi

# if [[ ${mod} == "train_mawps" ]];
# then

#     CUDA_VISIBLE_DEVICES=${device} python train_mawps.py \
#         --dataset_name 'mawps' \
#         --log_text ${log} \
#         --data_path '../data/MAWPS' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --save_model \
#         --cfg '{"model_name":"roberta-base","bert_lr":2e-5,"gru_lr":2e-5,"max_step_size":5,"save_result":true}'

#         # "model_name": "bert-base-uncased", "bert_lr": 5e-5, "gru_lr": 5e-4, "weight_decay": 1e-4
# fi

# if [[ ${mod} == "debug_mawps" ]];
# then

#     CUDA_VISIBLE_DEVICES=${device} python train_mawps.py \
#         --dataset_name 'mawps' \
#         --log_text '(debug)' \
#         --data_path '../data/MAWPS' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --cfg '{"model_name":"roberta-base","bert_lr":1e-5,"gru_lr":5e-4,"save_result":true,"max_step_size":7}' \
#         --head 1000 \
#         --debug \
#         --save_model

#         # "model_name": "bert-base-uncased", "bert_lr": 5e-5, "gru_lr": 5e-4, "weight_decay": 1e-4
# fi

# if [[ ${mod} == "debug_mathtoy" ]];
# then
#     CUDA_VISIBLE_DEVICES=${device} python train_mathtoy.py \
#         --dataset_name 'mathtoy' \
#         --log_text '(debug)' \
#         --data_path '../data/MathToy' \
#         --load_model_dir 'models_test' \
#         --save_model_dir 'models_test' \
#         --cfg '{"num_epochs":30}' \
#         --head 1000 \
#         --debug
# fi

# if [[ ${mod} == "debug_webqsp" ]];
# then

#     CUDA_VISIBLE_DEVICES=${device} python train_webqsp.py \
#         --dataset_name 'webqsp' \
#         --log_text '(debug)' \
#         --data_path '../data/WebQSP' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --cfg '{"num_epochs":30,"model_name":"bert-base-uncased"}' \
#         --debug

# fi