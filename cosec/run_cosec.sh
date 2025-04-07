#!/bin/bash

base_model=../trained/starcoderbase-7b_std/checkpoint-last
python train_lora_sec.py --base_model $base_model --output_name starcoderbase-7b-cosec --datasets sec-desc sec-new-desc