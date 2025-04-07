#!/bin/bash

name=stcd
neg=-1.0
pos=1.0
rank=1000
device='cuda:0'
model=starcoderbase-1b_std

python sec_eval.py --output_name $model-${name} --model_name $model-steer --eval_type trained --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python sec_eval.py --output_name $model-${name} --model_name $model-steer --eval_type trained-new --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}

python print_results.py --eval_name $model-${name} --eval_type trained-joint --detail

python sec_eval.py --output_name $model-${name} --model_name $model-steer --eval_type not-trained --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python print_results.py --eval_name $model-${name} --eval_type not-trained --detail

python func_eval_gen.py --eval_type human_eval --output_name $model-${name}-0.2 --model_name $model-steer --temp 0.2 --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python func_eval_exec.py --eval_type human_eval --output_name $model-${name}-0.2
python print_results.py --eval_name $model-${name}-0.2 --eval_type human_eval

python func_eval_gen.py --eval_type human_eval --output_name $model-${name}-0.6 --model_name $model-steer --temp 0.6 --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python func_eval_exec.py --eval_type human_eval --output_name $model-${name}-0.6
python print_results.py --eval_name $model-${name}-0.6 --eval_type human_eval

python func_eval_gen.py --eval_type mbpp --output_name $model-${name}-0.2 --model_name $model-steer --temp 0.2 --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python func_eval_exec.py --eval_type mbpp --output_name $model-${name}-0.2
python print_results.py --eval_name $model-${name}-0.2 --eval_type mbpp

python func_eval_gen.py --eval_type mbpp --output_name $model-${name}-0.6 --model_name $model-steer --temp 0.6 --num_steers 2 --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device ${device}
python func_eval_exec.py --eval_type mbpp --output_name $model-${name}-0.6
python print_results.py --eval_name $model-${name}-0.6 --eval_type mbpp
