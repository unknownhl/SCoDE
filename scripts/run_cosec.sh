#!/bin/bash

model=starcoderbase-1b_std
result=starcoderbase-1b
model_name_or_path=../trained/$model/checkpoint-last
base_model=../trained/$model/checkpoint-last
sec_model=../trained/$result-cosec/checkpoint-last

CUDA_VISIBLE_DEVICES=1,2 python sec_eval.py --output_name $result-cosec --model_name $result-cosec --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model --eval_type trained
CUDA_VISIBLE_DEVICES=1,2 python sec_eval.py --output_name $result-cosec --model_name $result-cosec --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model --eval_type trained-new
CUDA_VISIBLE_DEVICES=1,2 python sec_eval.py --output_name $result-cosec --model_name $result-cosec --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model --eval_type not-trained
python print_results.py --eval_name $result-cosec --eval_type trained-joint --detail

python print_results.py --eval_name $result-cosec --eval_type not-trained --detail

CUDA_VISIBLE_DEVICES=1,2 python func_eval_gen.py --eval_type human_eval --output_name $result-cosec-0.2 --model_name $result-cosec --temp 0.2 --exp_temp 0.2 --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model
CUDA_VISIBLE_DEVICES=1,2 python func_eval_exec.py --eval_type human_eval --output_name $result-cosec-0.2
python print_results.py --eval_name $result-cosec-0.2 --eval_type human_eval

CUDA_VISIBLE_DEVICES=1,2 python func_eval_gen.py --eval_type human_eval --output_name $result-cosec-0.6 --model_name $result-cosec --temp 0.6 --exp_temp 0.6 --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model
CUDA_VISIBLE_DEVICES=1,2 python func_eval_exec.py --eval_type human_eval --output_name $result-cosec-0.6
python print_results.py --eval_name $result-cosec-0.6 --eval_type human_eval

CUDA_VISIBLE_DEVICES=1,2 python func_eval_gen.py --eval_type mbpp --output_name $result-cosec-0.2 --model_name $result-cosec --temp 0.2 --exp_temp 0.2 --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model
CUDA_VISIBLE_DEVICES=1,2 python func_eval_exec.py --eval_type mbpp --output_name $result-cosec-0.2
python print_results.py --eval_name $result-cosec-0.2 --eval_type mbpp

CUDA_VISIBLE_DEVICES=1,2 python func_eval_gen.py --eval_type mbpp --output_name $result-cosec-0.6 --model_name $result-cosec --temp 0.6 --exp_temp 0.6 --model_name_or_path $model_name_or_path --base_model $base_model --sec_model $sec_model
CUDA_VISIBLE_DEVICES=1,2 python func_eval_exec.py --eval_type mbpp --output_name $result-cosec-0.6
python print_results.py --eval_name $result-cosec-0.6 --eval_type mbpp