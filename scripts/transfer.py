import os
import re
import sys

sys.path.append(os.path.join(os.getcwd(),".."))

import argparse
from tqdm import tqdm
import torch
from torch.optim import Adam
import shutil
from pathlib import Path

from steer.steer_models import Steer
from safecoder.utils import set_logging, set_seed 
from safecoder.constants import CWES_TRAINED, NEW_EVALS, NOT_TRAINED, PROMPT_NO_INPUT, INSTRUCTION
from safecoder.evaler import EvalerCodeSTEER
from safecoder.metric import FuncEval
from safecoder.human_eval.problem_yaml import Problem
from scripts.sec_eval import eval_all
from scripts.func_eval_exec import evaluate_problem


def get_transfer_model(args):
    if '1b' in args.model_name.lower():
        to_rank = 1000
    elif '3b' in args.model_name.lower():
        to_rank = 2000
    else:
        to_rank = 4000

    model = Steer(args, args.model_name, args.num_steers, to_rank, args.epsilon, args.init_var)
    to_ckpt_name = os.path.join(args.model_dir, args.model_name, 'checkpoint-last', 'pytorch_model.bin')
    model.load_state_dict(torch.load(to_ckpt_name, map_location = model.device))
    if not args.is_learn:
        from_ckpt_name = os.path.join(args.model_dir, args.transfer_from, 'checkpoint-last', 'pytorch_model.bin')
        projector1 = torch.nn.Parameter(torch.load(from_ckpt_name, map_location = model.device)["model.lm_head.projector1"].requires_grad_(True))
        projector2 = torch.nn.Parameter(torch.load(from_ckpt_name, map_location = model.device)["model.lm_head.projector2"].requires_grad_(True))
        print(projector1.shape, projector2.shape)
        print(model.model.lm_head.projector1.shape, model.model.lm_head.projector2.shape)
        model.model.lm_head.projector1 = projector1
        model.model.lm_head.projector2 = projector2
    else:
        print("loading learned weights")
        data = torch.load(args.output_file, map_location = model.device)
        projector1 = torch.nn.Parameter(data['projector1'].requires_grad_(True).to(model.device))
        projector2 = torch.nn.Parameter(data['projector2'].requires_grad_(True).to(model.device))
        model.model.lm_head.projector1 = projector1
        model.model.lm_head.projector2 = projector2
    
    return model

class Transfer(EvalerCodeSTEER):
    def __init__(self, args):
        self.args = args
        self.model = get_transfer_model(args)
        self.tokenizer = self.model.tokenizer


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

def get_embedding_vocab(args, model_name, is_from=False):
    if '1b' in model_name.lower():
        rank = 1000
    elif '3b' in model_name.lower():
        rank = 2000
    else:
        rank = 4000
    model = Steer(args, model_name, args.num_steers, rank, args.epsilon, args.init_var)
    tokenizer = model.tokenizer
    ckpt_name = os.path.join(args.model_dir, args.model_name, 'checkpoint-last', 'pytorch_model.bin')
    embeddings = torch.load(ckpt_name, map_location = model.device)["model.lm_head.lm_head.weight"]
    vocab = tokenizer.vocab
    if is_from:
        return embeddings, vocab
    else:
        projector1 = torch.load(ckpt_name, map_location = model.device)["model.lm_head.projector1"]
        projector2 = torch.load(ckpt_name, map_location = model.device)["model.lm_head.projector2"]
        return [projector1, projector2], embeddings, vocab


def train_transfer(args):
    args.logger.info(f'Training args {args}')
    device = torch.device(args.device)

    args.logger.info("starting to transfer")
    projectors, to_embeddings, to_vocab = get_embedding_vocab(args, args.model_name, is_from=False)
    from_embeddings, from_vocab = get_embedding_vocab(args, args.transfer_from, is_from=True)

    from_vocab_set = set(from_vocab.keys())
    shared_vocab = [
        _v for _v in tqdm(to_vocab.keys())
        if _v in from_vocab_set
    ]

    args.logger.info(f"from_embeddings shape: {from_embeddings.shape}")
    args.logger.info(f"to_embeddings shape: {to_embeddings.shape}")
    args.logger.info(f"from_vocab size: {len(from_vocab)}")
    args.logger.info(f"to_vocab size: {len(to_vocab)}")
    args.logger.info(f"shared vocab size: {len(shared_vocab)}")

    from_indices = [from_vocab[_v] for _v in shared_vocab]
    to_indices = [to_vocab[_v] for _v in shared_vocab]

    from_shared_embeddings = from_embeddings[from_indices].to(device).float()
    to_shared_embeddings = to_embeddings[to_indices].to(device).float()

    B_forward = torch.randn((from_embeddings.shape[1], to_embeddings.shape[1]),
                            requires_grad=True, device=device)
    B_backward = torch.randn((to_embeddings.shape[1],
                              from_embeddings.shape[1]),
                             requires_grad=True, device=device)
    optimizer = Adam([B_forward, B_backward], lr=args.lr)
    top_k = args.top_k

    pbar = tqdm(range(args.n_steps))
    for step_i in pbar:
        optimizer.zero_grad()
        loss1 = (from_shared_embeddings[:top_k].matmul(B_forward)
                 - to_shared_embeddings[:top_k]).pow(2).mean()
        loss2 = (to_shared_embeddings[:top_k].matmul(B_backward)
                 - from_shared_embeddings[:top_k]).pow(2).mean()

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if args.logging_steps > 0 and step_i % args.logging_steps == 0:
            args.logger.info(f"Step {step_i}: {loss.item()}")

    B_backward = B_backward / B_backward.norm(dim=1)[:, None]
    projector1 = B_backward.matmul(projectors[0].to(device))
    projector2 = B_backward.matmul(projectors[1].to(device))

    save_ckpt = {
        "projector1": projector1,
        "projector2": projector2
    }

    torch.save(save_ckpt, args.output_file)
    args.logger.info(f"Saved transfered weights to: {args.output_file}")


def sec_main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, os.path.join(args.output_dir, 'sec_eval.log'))
    set_seed(args.seed)
    args.logger.info(f'args: {args}')

    evaler = Transfer(args)

    if args.vul_type is not None:
        vul_types = [args.vul_type]
    elif args.eval_type == 'not-trained':
        vul_types = NOT_TRAINED
    elif args.eval_type == 'trained-new':
        vul_types = NEW_EVALS
    else:
        vul_types = CWES_TRAINED

    eval_all(args, evaler, vul_types)


def func_main(args):
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

    model = get_transfer_model(args)
    tokenizer = model.tokenizer
    model.eval()

    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()
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
            for sample in samples.tolist():
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion)
                completion = trim_code(completion, problem.stop_tokens)
                problem.completions.append(completion)
        with problem_yaml_path.open('w') as f:
            f.write(Problem.dump(problem))

def func_exec(args):
    files = [ p for p in Path(args.output_dir).glob("*.yaml") if not p.name.endswith(".results.yaml") ]
    for file in tqdm(files):
        evaluate_problem(file, args.max_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../experiments/transferred_weights')
    parser.add_argument('--model_name', type=str, default='starcoderbase-1b_std-steer')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)

    # steer
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=1000)
    parser.add_argument("--num_steers", type=int, default=2)
    parser.add_argument("--steer_values", default=None, nargs="*", type=float)
    parser.add_argument("--is_inference", action="store_true")

    # transfer related
    parser.add_argument("--transfer_from", type=str, default='starcoderbase-7b_std-steer')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=10000)
    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--is_learn', action="store_true")

    # sec eval related
    parser.add_argument('--is_sec', action="store_true")
    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained-new', 'not-trained', 'prompts', 'human_eval', 'mbpp'], default='trained')
    parser.add_argument('--sec_prompting', type=str, choices=['none', 'generic', 'specific'], default='none')
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_samples_per_gen', type=int, default=20)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/transferred')
    parser.add_argument('--data_dir', type=str, default='../data_eval/sec_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    # func exec related
    parser.add_argument("--max_workers", type=int, default=50)


    args = parser.parse_args()
    pattern = r"(.+)-(\d+)b"
    match = re.search(pattern, args.model_name)
    if match is None:
        raise ValueError("model name must be in the format of <model_name>-<num_layers>b")
    base_model_name = match.group(1)
    base_model_dir = os.path.join(args.output_dir, base_model_name)
    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir)
    transfer_direction = match.group(2)+'b_from_'+re.search(pattern, args.transfer_from).group(2)+'b'
    args.output_dir = os.path.join(base_model_dir, transfer_direction)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = os.path.join(args.output_dir, 'transfer.bin')
    if not os.path.exists(args.output_file):
        set_logging(args, os.path.join(args.output_dir, 'train.log'))
        set_seed(args.seed)
        train_transfer(args)
    else:
        if args.is_sec:
            assert args.num_samples % args.num_samples_per_gen == 0
            if args.model_name in ('octocoder', 'llama2-13b-chat', 'codellama-13b-chat'):
                args.num_samples_per_gen = 10
            args.output_dir = os.path.join(args.experiments_dir, base_model_name, transfer_direction + '-learned', args.eval_type)
            args.data_dir = os.path.join(args.data_dir, args.eval_type)
            sec_main(args)
        else:
            assert args.num_samples % args.num_samples_per_gen == 0

            args.output_dir = os.path.join(args.experiments_dir, base_model_name, transfer_direction + '-learned', args.eval_type)
            args.data_dir = '../data_eval'
            args.data_dir = os.path.join(args.data_dir, args.eval_type)
            os.makedirs(args.output_dir, exist_ok=True)
            args.output_dir = os.path.join(args.output_dir, str(args.temp))
            # if os.path.exists(args.output_dir):
            #     shutil.rmtree(args.output_dir)
            # shutil.copytree(args.data_dir, args.output_dir)
            # func_main(args)
            # func_exec(args)
            FuncEval(args.output_dir).pretty_print(detail=True)
