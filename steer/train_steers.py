import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(),".."))


from safecoder.utils import set_seed, set_logging
from steer.trainer_steers import Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--pretrain_name', type=str, default='codegen-350m')

    # sec and sec-instruct
    parser.add_argument('--loss_weight', type=float, default=1.0)

    # sven prefix-tuning
    parser.add_argument('--sven', action='store_true', default=False)

    # training arguments 
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_acc_steps', type=int, default=16)
    parser.add_argument('--exclude_neg', action='store_true', default=False)
    parser.add_argument('--no_weights', action='store_true', default=False)

    parser.add_argument('--kl_loss_weight', type=int, default=0) # will be divided by 1000

    # upsampling arguments
    """
    --sampling_size:
        the size of sampling, <=0 means no sampling
        dataset.Upsampler._upsample_all_prob: the percentange of the sampled sec dataset compared to the func dataset
        dataset.Upsampler._upsample_minority: sample classes with <sampling_size samples to sampling_size
    --sampling_weight:
        select the mode of how the sampling weight of each cwe at lang is calcualted when doing -all sempling modes:
            uniform: each example is treated equally and uniform sampling is employed across the whole dataset
            inverse-prop: cwes with less example are more likely to get sampled, chosing this balances the cwes
    --cwes:
        select a list of the cwes you want to upsample, or select all
    --langs:
        select a list of the langs you want to include in upsampling, or select all
    """
    parser.add_argument('--sampling_size', type=int, default=-1)
    parser.add_argument('--sampling_method', type=str, choices=['uniform', 'inverse-prop', 'minority'], default='minority')
    parser.add_argument('--cwes', type=str, nargs='*', default=['all'])
    parser.add_argument('--langs', type=str, nargs='*', default=['all'])

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--model_dir', type=str, default='../trained/')


    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=1000)
    parser.add_argument("--num_steers", type=int, default=10)
    parser.add_argument("--temperature", type=int, default=1)

    # Training related
    parser.add_argument("--regularization", type=float, default=0)
    parser.add_argument("--gamma_mean", type=float, default=0.99)
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dummy_steer", type=int, default=None)
    parser.add_argument("--training_steer", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--is_inference", action="store_true")


    args = parser.parse_args()

    # adjust the naming to make sure that it is in the expected format for loading

    if args.sampling_size == -1 and 'lmsys' in args.datasets:
        args.sampling_size = 40

    if args.sampling_size == -1 and 'evol' in args.datasets:
        args.sampling_size = 20


    args.num_train_epochs = 5

    if args.learning_rate is None:
        if args.sven:
            if args.pretrain_name.startswith('starcoderbase'):
                args.learning_rate = 5e-2
            else:
                args.learning_rate = 1e-2
        else:
            args.learning_rate = 1e-3

    if args.exclude_neg:
        args.sampling_size = args.sampling_size // 2

    args.output_dir = os.path.join(args.model_dir, args.output_name)

    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_seed(args.seed)
    Trainer(args).run()

if __name__ == '__main__':
    main()