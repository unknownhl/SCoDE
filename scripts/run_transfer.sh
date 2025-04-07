#!/bin/bash

name=steer
neg=-1.0
pos=1.0
# neg=-0.5
# pos=0.5
rank=1000
device=0
to_model=starcoderbase-1b_std
from_model=starcoderbase-7b_std
# to_model=qwen2.5-coder-3b_std
# from_model=qwen2.5-coder-1b_std

# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type trained  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type trained-new --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type not-trained --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# python print_results.py --eval_name ${to_model} --eval_type transfer


# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type human_eval  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.2 --num_samples 40 --num_samples_per_gen 10
# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type human_eval  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.6 --num_samples 40 --num_samples_per_gen 10
# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type mbpp  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.2 --num_samples 40 --num_samples_per_gen 10
# CUDA_VISIBLE_DEVICES=$device python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type mbpp  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.6 --num_samples 40 --num_samples_per_gen 10


# python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type trained  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type trained-new --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type not-trained --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --is_sec
# python print_results.py --eval_name ${to_model} --eval_type transfer


python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type human_eval  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.2 --num_samples 40 --num_samples_per_gen 10
python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type human_eval  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.6 --num_samples 40 --num_samples_per_gen 10
python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type mbpp  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.2 --num_samples 40 --num_samples_per_gen 10
python transfer.py --model_name ${to_model}-steer --transfer_from ${from_model}-steer --eval_type mbpp  --rank ${rank} --steer_values ${neg} ${pos} --is_inference --device cuda:${device} --is_learn --temp 0.6 --num_samples 40 --num_samples_per_gen 10