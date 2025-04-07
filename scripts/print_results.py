import os
import re
import sys
sys.path.append(os.path.join(os.getcwd(),".."))

import json
import argparse

from safecoder.metric import FuncEval, SecEval, MMLUEval, TruthfulQAEval

EVAL_CHOICES = [
    'human_eval',
    'mbpp',
    'trained',
    'trained-new',
    'trained-joint',
    'not-trained',
    'mmlu',
    'tqa',
    'transfer',
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True)
    parser.add_argument('--detail', action='store_true', default=False)
    parser.add_argument('--eval_type', type=str, choices=EVAL_CHOICES, default='trained')
    parser.add_argument('--split', type=str, choices=['val', 'test', 'all', 'validation', 'intersec', 'diff'], default='test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    args = parser.parse_args()

    if args.n_shots is None:
        if args.eval_type == 'multiple_choice':
            args.n_shots = 5
        elif args.eval_type == 'mmlu':
            args.n_shots = 5

    return args

def main():
    args = get_args()
    if args.eval_type in ('human_eval', 'mbpp'):
        e = FuncEval(os.path.join(args.experiments_dir, args.eval_type, args.eval_name))
    elif args.eval_type == 'mmlu':
        e = MMLUEval(os.path.join(
            args.experiments_dir, 'mmlu_eval', args.eval_name, args.eval_type, args.split, f'result_{args.n_shots}_{args.seed}.csv'
        ))
    elif args.eval_type == 'tqa':
        e = TruthfulQAEval(os.path.join(
            args.experiments_dir, 'truthfulqa_eval', args.eval_name, 'multiple_choice', args.split, f'result_{args.n_shots}_{args.seed}.csv'
        ))
    elif args.eval_type == 'transfer':
        pattern = r"(.+)-(\d+)b"
        match = re.search(pattern, args.eval_name)
        if match is None:
            raise ValueError("model name must be in the format of <model_name>-<num_layers>b")
        base_model_name = match.group(1)
        if match.group(2) == '1' and 'qwen' in base_model_name:
            transfer_direction = '1b_from_3b-learned'
        if match.group(2) == '3' and 'qwen' in base_model_name:
            transfer_direction = '3b_from_1b-learned'
        if match.group(2) == '1' and 'starcoder' in base_model_name:
            transfer_direction = '1b_from_7b-learned'
        if match.group(2) == '7' and 'starcoder' in base_model_name:
            transfer_direction = '7b_from_1b-learned'
        for t in ['trained','trained-new','trained-joint','not-trained']:
            print(f"===================={t}====================")
            e = SecEval(os.path.join(args.experiments_dir, 'transferred', base_model_name, transfer_direction), args.split, t)
            e.pretty_print(args.detail)
    else:
        e = SecEval(os.path.join(args.experiments_dir, 'sec_eval', args.eval_name), args.split, args.eval_type)
    e.pretty_print(args.detail)

if __name__ == '__main__':
    main()