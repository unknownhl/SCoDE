#!/bin/bash

python train.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-steer --datasets evol sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3

python train_std.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-steer_std --datasets evol --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3

python train_sec.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-steer_sec --datasets sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3


python train_std.py --pretrain_name codellama-7b --output_name codellama-7b-steer_std --datasets evol --lora --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3


python train_sec.py --pretrain_name codellama-7b --output_name codellama-7b-steer_sec --datasets sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3
python train_sec.py --pretrain_name codellama-7b-steer_std --output_name codellama-7b-steer_sec --datasets sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3

# train safecoder lora
python train.py --pretrain_name codellama-7b --output_name codellama-7b-lora-safecoder --datasets evol sec-desc sec-new-desc --lora

# train std
python train.py --pretrain_name codellama-7b --output_name codellama-7b_std --datasets evol --lora

# train sven
python train.py --pretrain_name codellama-7b_std --output_name codellama-7b_std-sven --datasets sec-desc sec-new-desc --sven --kl_loss_weight 1600

# latest
python train_final.py --pretrain_name deepseek_std --output_name deepseek_std-steer --datasets sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3