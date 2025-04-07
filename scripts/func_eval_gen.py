import os
import re
import sys

sys.path.append(os.path.join(os.getcwd(),".."))

import torch
import torch.nn as nn
import numpy
import random
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cosec.CustomizedGeneration import CodeLlamaModelLM, StarcodeModelLM, CodegenModelLM, Qwen2ModelLM

from safecoder.utils import set_seed, load_model
from safecoder.human_eval.problem_yaml import Problem
from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS, PROMPT_NO_INPUT, INSTRUCTION

from steer.steer_models import Steer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, required=True, choices=['human_eval', 'mbpp'])
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='codegen-350m')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=40)
    parser.add_argument('--num_samples_per_gen', type=int, default=10)

    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    # steer
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=1000)
    parser.add_argument("--num_steers", type=int, default=2)
    parser.add_argument("--steer_values", default=None, nargs="*", type=float)
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--is_inference", action="store_true")

    parser.add_argument('--text_contrast', action="store_true")
    parser.add_argument('--wo_st', action="store_true")

    # cosec
    parser.add_argument('--model_name_or_path', type=str, default='', help='your target model path')
    parser.add_argument('--base_model', type=str, default='', help='base model of your security model')
    parser.add_argument('--sec_model', type=str, default='', help='lora part of your security model')
    parser.add_argument('--exp_temp', type=float, default=0.4)
    parser.add_argument('--threshold', type=float, default=0.3)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.experiments_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args

args = get_args()

def extract_docstr(prompt):
    delim = '\"\"\"'
    assert delim in prompt

    output = prompt[prompt.find(delim)+len(delim):]
    output = output[:output.find(delim)]
    output = output.replace('\n    ', '\n').strip()

    return output

def extract_funcsig(prompt):
    delim = '\"\"\"'
    return prompt[:prompt.find(delim)].strip()

def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion

def _get_sec_model(args):
    if 'deepseek' in args.model_name_or_path:
        model = CodeLlamaModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'codellama' in args.model_name_or_path:
        model = CodeLlamaModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'qwen' in args.model_name_or_path:
        model = Qwen2ModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'star' in args.model_name_or_path:
        model = StarcodeModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    elif 'codegen' in args.model_name_or_path:
        model = CodegenModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
        sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    else:
        raise NotImplementedError()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.eval()
    sec_model.eval()
    return model, sec_model, tokenizer

def main():
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    if 'steer' in args.model_name:
        model = Steer(args, args.model_name, args.num_steers, args.rank, args.epsilon, args.init_var)
        tokenizer = model.tokenizer
        ckpt_name = os.path.join(args.model_dir, args.model_name, 'checkpoint-last', 'pytorch_model.bin')
        model.load_state_dict(torch.load(ckpt_name, map_location = model.device))
        model.eval()

        if args.wo_st:
            embed_dim = model.get_embed_dim()
            # random initialization
            projector1 = nn.Parameter(torch.randn(
                args.num_steers, embed_dim, args.rank
            ) * args.init_var).to(model.device)
            projector2 = nn.Parameter(torch.randn(
                args.num_steers, embed_dim, args.rank
            ) * args.init_var).to(model.device)
            model.model.lm_head.projector1.data = projector1.data
            model.model.lm_head.projector2.data = projector2.data

    elif 'cosec' in args.model_name:
        model, sec_model, tokenizer = _get_sec_model(args)
    else:
        tokenizer, model = load_model(args.model_name, args)
        model.eval()
    is_pretrained = args.model_name in PRETRAINED_MODELS
    is_chat = args.model_name in CHAT_MODELS

    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()
        if is_chat:
            if args.model_name == 'octocoder':
                template = 'Question: {instruction}\n\nAnswer: '
                prompt = template.format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
                prompt += extract_funcsig(orig_prompt)
            else:
                prompt = PROMPT_NO_INPUT[:PROMPT_NO_INPUT.rfind('\n\n')].format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
                messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': extract_funcsig(orig_prompt)}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt = prompt.removeprefix('<s>').removesuffix('</s> ').removesuffix(' </s>')
        elif is_pretrained:
            prompt = orig_prompt
    
        else:
            prompt = PROMPT_NO_INPUT.format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
            prompt += extract_funcsig(orig_prompt)
        
        inputs = tokenizer(prompt.strip(), return_tensors='pt').to(model.device)
        seed = args.seed
        for i in range(args.num_samples // args.num_samples_per_gen):
            set_seed(seed+i)
            with torch.no_grad():

                if hasattr(model.config, 'n_positions'):
                    n_ctx = model.config.n_positions
                elif hasattr(model.config, 'max_position_embeddings'):
                    n_ctx = model.config.max_position_embeddings
                else:
                    n_ctx = 32000 # some arbitrary large context, risky as it could lead to errors
                max_gen_len = max(0, min(n_ctx - 1 - len(inputs['input_ids'][0]), args.max_gen_len))
                if 'steer' in args.model_name:
                    samples = model.generate(
                        inputs['input_ids'],
                        steer_values=list(map(float, args.steer_values)) if args.steer_values is not None else None,
                        do_sample=True,
                        num_return_sequences=args.num_samples_per_gen,
                        temperature=args.temp,
                        max_new_tokens=max_gen_len,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                elif 'cosec' in args.model_name:
                    kwargs = {
                        'expert': True,
                        'expert_lm': sec_model,
                        'model_kwargs_expert': {},
                        'threshold': args.threshold,
                    }
                    samples = model.generate_with_experts(
                        inputs['input_ids'],
                        do_sample=True,
                        num_return_sequences=args.num_samples_per_gen,
                        temperature=args.temp,
                        max_new_tokens=max_gen_len,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        expert_min_prob=0.0,
                        expert_temperature=args.exp_temp,
                        expert_top_p=0.95,
                        **kwargs
                    )
                else:
                    samples = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=args.num_samples_per_gen,
                        temperature=args.temp,
                        max_new_tokens=max_gen_len,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
            for sample in samples.tolist():
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion)
                completion = trim_code(completion, problem.stop_tokens)
                problem.completions.append(completion)
        with problem_yaml_path.open('w') as f:
            f.write(Problem.dump(problem))

if __name__ == '__main__':
    main()