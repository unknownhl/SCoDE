#!/bin/bash

device=0
model=starcoderbase-1b_std

CUDA_VISIBLE_DEVICES=$device python sec_eval.py --output_name $model-sven --model_name $model-sven --eval_type trained
CUDA_VISIBLE_DEVICES=$device python sec_eval.py --output_name $model-sven --model_name $model-sven --eval_type trained-new
CUDA_VISIBLE_DEVICES=$device python sec_eval.py --output_name $model-sven --model_name $model-sven --eval_type not-trained
python print_results.py --eval_name $model-sven --eval_type trained-joint --detail

python print_results.py --eval_name $model-sven --eval_type not-trained --detail

CUDA_VISIBLE_DEVICES=$device ./func_eval.sh human_eval $model-sven-0.2 $model-sven 0.2
python print_results.py --eval_name $model-sven-0.2 --eval_type human_eval

CUDA_VISIBLE_DEVICES=$device ./func_eval.sh human_eval $model-sven-0.6 $model-sven 0.6
python print_results.py --eval_name $model-sven-0.6 --eval_type human_eval

CUDA_VISIBLE_DEVICES=$device ./func_eval.sh mbpp $model-sven-0.2 $model-sven 0.2
python print_results.py --eval_name $model-sven-0.2 --eval_type mbpp

CUDA_VISIBLE_DEVICES=$device ./func_eval.sh mbpp $model-sven-0.6 $model-sven 0.6
python print_results.py --eval_name $model-sven-0.6 --eval_type mbpp
