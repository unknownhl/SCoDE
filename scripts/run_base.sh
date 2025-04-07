#!/bin/bash

model=starcoderbase-1b
CUDA_VISIBLE_DEVICES=1 python sec_eval.py --output_name $model --model_name $model --eval_type trained
CUDA_VISIBLE_DEVICES=1 python sec_eval.py --output_name $model --model_name $model --eval_type trained-new
CUDA_VISIBLE_DEVICES=1 python sec_eval.py --output_name $model --model_name $model --eval_type not-trained
python print_results.py --eval_name $model --eval_type trained-joint --detail


# HumanEval, with temperature 0.2
CUDA_VISIBLE_DEVICES=1 ./func_eval.sh human_eval $model-0.2 $model 0.2
python print_results.py --eval_name $model-0.2 --eval_type human_eval

CUDA_VISIBLE_DEVICES=1 ./func_eval.sh human_eval $model-0.6 $model 0.6
python print_results.py --eval_name $model-0.6 --eval_type human_eval

# MBPP, with temperature 0.2
CUDA_VISIBLE_DEVICES=1 ./func_eval.sh mbpp $model-0.2 $model 0.2
python print_results.py --eval_name $model-0.2 --eval_type mbpp

CUDA_VISIBLE_DEVICES=1 ./func_eval.sh mbpp $model-0.6 $model 0.6
python print_results.py --eval_name $model-0.6 --eval_type mbpp
