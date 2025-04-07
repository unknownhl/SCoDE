import argparse
import logging
import os
from collections import OrderedDict
import sys
sys.path.append('../')
from utils import add_to_loss_dict, report_loss_dict, save

import torch.nn.functional as F
import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import set_seed, AutoTokenizer, CodeGenForCausalLM, default_data_collator, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM

from safecoder.dataset import CodeDataset

logger = logging.getLogger()
def parse_args():

    parser = argparse.ArgumentParser(description='Codegen LoRA for security generation')

    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--base_model', type=str, default='../codegen-350M', help='模型id或local path')

    # sec and sec-instruct
    parser.add_argument('--loss_weight', type=float, default=1.0)

    # sven prefix-tuning
    parser.add_argument('--sven', action='store_true', default=False)

    parser.add_argument('--num_train_epochs', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    # parser.add_argument('--num_workers_dataloader', type=int, default=1)
    parser.add_argument('--lora_rank', type=int, default=8, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')

    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default='resume/', help='恢复训练的checkpoint路径')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--exclude_neg', action='store_true', default=False)
    parser.add_argument('--no_weights', action='store_true', default=False)

    parser.add_argument('--sampling_size', type=int, default=-1)
    parser.add_argument('--sampling_method', type=str, choices=['uniform', 'inverse-prop', 'minority'], default='minority')
    parser.add_argument('--cwes', type=str, nargs='*', default=['all'])
    parser.add_argument('--langs', type=str, nargs='*', default=['all'])

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--model_dir', type=str, default='../trained/')


    return parser.parse_args()

def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    args.logger = logger


def set_devices(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.logger.info('Device: %s, n_gpu: %s', device, args.n_gpu)

def do_eval(args, model, ref_model, eval_dataloader, tokenizer):
    acc_loss_dict = OrderedDict()
    for batch in eval_dataloader:
        return_dict = OrderedDict()

        sample_types, inputs, weights = batch
        inputs = inputs.to(args.device)
        weights = weights.to(args.device)

        shift_inputs = inputs[..., 1:].squeeze(0)
        shift_weights = weights[..., 1:].squeeze(0)

        outputs = model(inputs)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = inputs[..., 1:].unsqueeze(-1)
        shift_probs = F.softmax(shift_logits, dim=-1)

        correct_logits = shift_logits.squeeze(0)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(correct_logits, shift_inputs)
        lm_loss = lm_loss[shift_weights != 0]
        lm_loss = lm_loss.mean()
        return_dict['lm_loss'] = lm_loss.item()

        kl_loss = 0
        shift_weights_ = 1 - shift_weights
        correct_log_probs = F.log_softmax(correct_logits, dim=-1)
        with torch.no_grad():
            ref_outputs = ref_model(inputs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :]
            ref_logits = ref_shift_logits.squeeze(0)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        kl_loss_ = loss_fct(correct_log_probs, ref_log_probs)
        kl_loss_ = kl_loss_.sum(dim=1)
        kl_loss_ = kl_loss_[shift_weights_ != 0]
        kl_loss_ = kl_loss_.mean()
        kl_loss += kl_loss_
        kl_loss = kl_loss
        return_dict['kl_loss'] = kl_loss.item()

        loss = lm_loss + kl_loss

        return_dict['loss'] = loss.item()
        add_to_loss_dict(acc_loss_dict, return_dict)
    return report_loss_dict(acc_loss_dict, len(eval_dataloader))

def train(args, model, ref_model, train_dataloader, val_dataloader, tokenizer):

    total_samples = len(train_dataloader)
    batch_size = args.batch_size * args.grad_acc_steps
    total_steps = total_samples // batch_size * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    args.logger.info('***** Running training *****')
    args.logger.info('  Num samples = %d', total_samples)
    args.logger.info('  Num epoch = %d', args.num_train_epochs)
    args.logger.info('  Batch size= 1')
    args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
    args.logger.info('  Gradient Accumulation steps = %d', args.grad_acc_steps)
    args.logger.info('  Total optimization steps = %d', total_steps)
    args.logger.info('  Num val samples = %d', len(val_dataloader))
    model.print_trainable_parameters()

    global_step, acc_loss_dict = 0, OrderedDict()
    for epoch in range(args.num_train_epochs):
        model.train()
        ref_model.eval()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            return_dict = OrderedDict()
            sample_types, inputs, weights = batch
            inputs = inputs.to(args.device)
            weights = weights.to(args.device)


            shift_inputs = inputs[..., 1:].squeeze(0)
            shift_weights = weights[..., 1:].squeeze(0)

            outputs = model(inputs)
            shift_logits = outputs.logits[..., :-1, :]

            correct_logits = shift_logits.squeeze(0)

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            lm_loss = loss_fct(correct_logits, shift_inputs)
            lm_loss = lm_loss[shift_weights != 0]
            lm_loss = lm_loss.mean()
            return_dict['lm_loss'] = lm_loss.item()


            kl_loss = 0
            shift_weights_ = 1 - shift_weights
            correct_log_probs = F.log_softmax(correct_logits, dim=-1)
            with torch.no_grad():
                ref_outputs = ref_model(inputs)
                ref_shift_logits = ref_outputs.logits[..., :-1, :]
                ref_logits = ref_shift_logits.squeeze(0)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
            kl_loss_ = loss_fct(correct_log_probs, ref_log_probs)
            kl_loss_ = kl_loss_.sum(dim=1)
            kl_loss_ = kl_loss_[shift_weights_ != 0]
            kl_loss_ = kl_loss_.mean()
            kl_loss += kl_loss_
            kl_loss = kl_loss
            return_dict['kl_loss'] = kl_loss.item()

            loss = lm_loss + kl_loss
            return_dict['loss'] = loss.item()

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
                for key in  return_dict:
                    return_dict[key] =  return_dict[key] / args.grad_acc_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            add_to_loss_dict(acc_loss_dict, return_dict)

            if (step + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    reported_loss = report_loss_dict(acc_loss_dict, args.logging_steps)
                    args.logger.info('epochs: %s/%d, steps: %s/%d, %s', epoch + 1, int(args.num_train_epochs),
                                          global_step, total_steps, reported_loss)
                    acc_loss_dict.clear()

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            model.eval()
            with torch.no_grad():
                reported_eval_loss = do_eval(args, model, ref_model, val_dataloader, tokenizer)
            model.train()
            args.logger.info('val epoch %s: %s', epoch + 1, reported_eval_loss)
            output_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch + 1}')
            last_output_dir = os.path.join(args.output_dir, f'checkpoint-last')
            args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            save(output_dir, model, tokenizer, global_step, epoch + 1, None, None, args)
            save(last_output_dir, model, tokenizer, global_step, epoch + 1, None, None, args)

    if (epoch+1) % args.save_epochs != 0:
        model.eval()
        with torch.no_grad():
            eval_loss_pp = do_eval(args, model, ref_model, val_dataloader, tokenizer)
        args.logger.info('final eval loss: %s', eval_loss_pp)
        last_output_dir = os.path.join(args.output_dir, f'checkpoint-last')
        args.logger.info('Saving model checkpoint to %s', last_output_dir)
        save(last_output_dir, model, tokenizer, global_step, epoch + 1, None, None, args)


def main():

    args = parse_args()
    args.output_dir = os.path.join(args.model_dir, args.output_name)

    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='cuda:1', )

    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='cuda:1', )
    # if 'codellama' in args.base_model:
    #     model.resize_token_embeddings(len(tokenizer))
    #     ref_model.resize_token_embeddings(len(tokenizer))
    #     model = PeftModel.from_pretrained(model, args.base_model)
    #     ref_model = PeftModel.from_pretrained(ref_model, args.base_model)
    #     model = model.merge_and_unload()
    #     ref_model = ref_model.merge_and_unload()
    # else:
    #     model.resize_token_embeddings(len(tokenizer))
    #     ref_model.resize_token_embeddings(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))

    if 'starcoder' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['gpt_bigcode']
    elif 'codellama' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
    elif 'deepseek' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
    elif 'qwen' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['qwen2']
    elif 'phi' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']
    elif 'mistral' in args.base_model:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['mistral']
    else:
        AssertionError(f'Unknown model {args.base_model}')

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')


    train_dataset = CodeDataset(args, tokenizer, 'train')
    val_dataset = CodeDataset(args, tokenizer, 'val')

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=1)

    train(args, model, ref_model, train_dataloader, val_dataloader, tokenizer)


if __name__ == '__main__':
    main()