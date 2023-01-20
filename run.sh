#! /bin/bash

set -e

mod=$1
log=$2
device=$3

if [ -z $3 ]; then
    device=0
fi

echo "mod: ${mod}"
echo "device: ${device}"

cd src

if [[ ${mod} == "train_svamp" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --model_type 're' \
        --dataset_name 'svamp' \
        --log_text ${log} \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"max_step_size":5,"num_epochs":1000,"batch_size":8}'
fi
# use_data_aug true

if [[ ${mod} == "debug_svamp" ]];
then

    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${device} python train_svamp.py \
        --model_type 're' \
        --dataset_name 'svamp' \
        --log_text '(debug)' \
        --data_path '../data/SVAMP' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"max_step_size":5,"save_result":true,"batch_size":8}' \
        --head 1000 \
        --debug

fi

if [[ ${mod} == "train_mawps" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mawps.py \
        --model_type 're' \
        --dataset_name 'mawps' \
        --log_text ${log} \
        --data_path '../data/MAWPS' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"max_step_size":5,"save_result":false,"num_epochs":500}'

fi

if [[ ${mod} == "debug_mawps" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mawps.py \
        --model_type 're' \
        --dataset_name 'mawps' \
        --log_text '(debug)' \
        --data_path '../data/MAWPS' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"max_step_size":5,"save_result":false,"num_epochs":500}' \
        --head 1000 \
        --debug

fi

if [[ ${mod} == "train_mathqa" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mathqa.py \
        --model_type 're' \
        --dataset_name 'mathqa' \
        --log_text ${log} \
        --data_path '../data/MathQA' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"save_result":true,"num_epochs":200,"save_result":true}'

fi

if [[ ${mod} == "test_mathqa" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python test_mathqa.py \
        --model_type 're' \
        --dataset_name 'mathqa' \
        --log_text ${log} \
        --data_path '../data/MathQA' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","beam_size":4}'

fi


if [[ ${mod} == "debug_mathqa" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mathqa.py \
        --model_type 're' \
        --dataset_name 'mathqa' \
        --log_text '(debug)' \
        --data_path '../data/MathQA' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"save_result":false,"num_epochs":200}' \
        --head 100 \
        --debug

fi

if [[ ${mod} == "train_math23k" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
        --model_type 're' \
        --dataset_name 'math23k' \
        --log_text ${log} \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":200,"lr":2e-5,"lr_alpha":1.0,"save_result":true}'

fi


if [[ ${mod} == "test_math23k" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python test_math23k.py \
        --model_type 're' \
        --dataset_name 'math23k' \
        --log_text ${log} \
        --data_path '../data/Math23K' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","beam_size":4}'

fi


# debug
if [[ ${mod} == "debug_math23k" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_math23k.py \
        --model_type 're' \
        --dataset_name 'math23k' \
        --log_text '(debug)' \
        --data_path '../data/Math23K' \
        --load_model_dir 'models_test' \
        --save_model_dir 'models_test' \
        --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":200,"lr":2e-5,"lr_alpha":1.0}' \
        --head 1000 \
        --debug
fi

if [[ ${mod} == "train_math23k_5fold" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_math23k_5fold.py \
        --model_type 're' \
        --dataset_name 'math23k' \
        --log_text ${log} \
        --data_path '../data/Math23K-5fold' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":200,"lr":2e-5,"lr_alpha":1.0}'

fi

if [[ ${mod} == "train_math23k_raw" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_math23k_raw.py \
        --model_type 're' \
        --dataset_name 'math23k' \
        --log_text ${log} \
        --data_path '../data/Math23K-raw' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":200,"lr":2e-5}' \
        --save_model \
        --expr_mode v3

fi


if [[ ${mod} == "train_template" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_template.py \
        --dataset_name 'template' \
        --log_text ${log} \
        --data_path '../data/MathTemplate' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","num_epochs":200,"lr":2e-5,"quant_size":100,"save_result":false}'

fi

# debug
if [[ ${mod} == "debug_template" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_template.py \
        --dataset_name 'template' \
        --log_text '(debug)' \
        --data_path '../data/MathTemplate' \
        --load_model_dir 'models' \
        --save_model_dir 'models' \
        --cfg '{"model_name":"roberta-base","num_epochs":200,"lr":2e-5,"quant_size":100,"save_result":false}' \
        --head 100 \
        --debug
fi

if [[ ${mod} == "train_template_binary" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_template_binary.py \
        --dataset_name 'templateBinary' \
        --log_text ${log} \
        --data_path '../data/MathTemplateBinary' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","num_epochs":200,"lr":2e-5,"quant_size":100,"save_result":false}'
fi

# debug
if [[ ${mod} == "debug_template_binary" ]];
then
    CUDA_VISIBLE_DEVICES=${device} python train_template_binary.py \
        --dataset_name 'templateBinary' \
        --log_text '(debug)' \
        --data_path '../data/MathTemplateBinary' \
        --load_model_dir 'models' \
        --save_model_dir 'models' \
        --cfg '{"model_name":"roberta-base","num_epochs":200,"lr":2e-5,"quant_size":100,"save_result":false}' \
        --head 100 \
        --debug
fi


if [[ ${mod} == "train_mathqa_abl0" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mathqa.py \
        --model_type 'rpe_abl0' \
        --dataset_name 'mathqa' \
        --log_text ${log} \
        --data_path '../data/MathQA' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --save_model \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"save_result":false,"num_epochs":200}'

fi

if [[ ${mod} == "debug_mathqa_abl0" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python train_mathqa.py \
        --model_type 'rpe_abl0' \
        --dataset_name 'mathqa' \
        --log_text '(debug)' \
        --data_path '../data/MathQA' \
        --load_model_dir '../models' \
        --save_model_dir '../models' \
        --cfg '{"model_name":"roberta-base","lr":2e-5,"save_result":false,"num_epochs":200}' \
        --head 100 \
        --debug

fi

if [[ ${mod} == "compute_params" ]];
then

    CUDA_VISIBLE_DEVICES=${device} python compute_params.py \
        --model_type 're' \
        --cfg '{"model_name":"roberta-base"}' \

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
#         --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":100}' \
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
#         --cfg '{"model_name":"hfl/chinese-roberta-wwm-ext","num_epochs":100}' \
#         --head 1000 \
#         --expr_mode v3 \
#         --debug
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

# if [[ ${mod} == "debug_webqsp" ]];
# then

#     CUDA_VISIBLE_DEVICES=${device} python train_webqsp.py \
#         --dataset_name 'webqsp' \
#         --log_text '(debug)' \
#         --data_path '../data/WebQSP' \
#         --load_model_dir '../models' \
#         --save_model_dir '../models' \
#         --cfg '{"model_name":"roberta-base","num_epochs":30}' \
#         --debug

# fi
