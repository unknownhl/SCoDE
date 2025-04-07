#!/bin/bash

model=qwen2.5-coder-3b
python sec_eval.py --output_name $model-safecoder --model_name $model-safecoder --eval_type trained
python sec_eval.py --output_name $model-safecoder --model_name $model-safecoder --eval_type trained-new
python sec_eval.py --output_name $model-safecoder --model_name $model-safecoder --eval_type not-trained
python print_results.py --eval_name $model-safecoder --eval_type trained-joint --detail


# HumanEval, with temperature 0.2
./func_eval.sh human_eval $model-safecoder-0.2 $model-safecoder 0.2
python print_results.py --eval_name $model-safecoder-0.2 --eval_type human_eval

./func_eval.sh human_eval $model-safecoder-0.6 $model-safecoder 0.6
python print_results.py --eval_name $model-safecoder-0.6 --eval_type human_eval

# MBPP, with temperature 0.2
./func_eval.sh mbpp $model-safecoder-0.2 $model-safecoder 0.2
python print_results.py --eval_name $model-safecoder-0.2 --eval_type mbpp

./func_eval.sh mbpp $model-safecoder-0.6 $model-safecoder 0.6
python print_results.py --eval_name $model-safecoder-0.6 --eval_type mbpp
