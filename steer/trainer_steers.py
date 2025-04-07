import os
import re
import sys

sys.path.append(os.path.join(os.getcwd(),".."))

import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from safecoder.utils import set_seed
from safecoder.timer import Timer
from safecoder.dataset import CodeDataset
from safecoder.constants import FUNC, GOOD, BAD

from steer.steer_models import Steer
from steer.steer_utils import RunningMean

class LossDict:
    def __init__(self, keys):
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps
                p.append(f'{k}: {round(s, 6)}')
        return ', '.join(p)

    def clear(self):
        for key in self.keys:
            self.d[key].clear()

    def __getitem__(self, k):
        return self.d[k]

def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'ce':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'nll':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.NLLLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'ul':
        probs = F.softmax(inputs, dim=-1)
        probs = torch.gather(probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.clamp((1.0-probs), min=1e-5)
        loss = -torch.log(probs)
    elif loss_type == 'kl':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        weights = weights.view(-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(inputs, targets)
        loss = loss.sum(dim=1)
    else:
        assert False

    loss = loss[weights != 0]
    return loss.mean()


def get_logits_from_lm(lm, inputs, steers):
    outputs = lm(inputs, steer_values = steers)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.loss_keys = ['lm', 'contra']
    
    def step(self, batch):
        loss_dict = LossDict(self.loss_keys)

        control_ids, inputs, weights = batch
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:].squeeze(0)

        correct_stance = torch.zeros(len(control_ids), self.args.num_steers).to(self.model.device)
        correct_labels = [it if it != BAD else -1 for it in control_ids] # BAD -> -1, GOOD -> 1
        correct_stance[:, self.args.training_steer] = torch.Tensor(correct_labels).to(self.model.device)
        if self.args.dummy_steer is not None:
            correct_stance[:, self.args.dummy_steer] = 1

        correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs, correct_stance)
        lm_loss = token_weighted_loss('ce', correct_logits, shift_inputs, shift_weights)
        loss_dict['lm'].append(lm_loss.item())

        incorrect_labels = [-1 * it for it in correct_labels]
        # incorrect_labels = [0 for it in control_ids]
        incorrect_stance = torch.zeros(len(control_ids), self.args.num_steers).to(self.model.device)
        incorrect_stance[:, self.args.training_steer] = torch.Tensor(incorrect_labels).to(self.model.device)
        if self.args.dummy_steer is not None:
            incorrect_stance[:, self.args.dummy_steer] = 1
        incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs, incorrect_stance)

        contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
        contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
        contrastive_log_probs = torch.log(contrastive_probs)
        contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64).to(self.model.device)
        contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
        contrastive_loss *= 4
        loss_dict['contra'].append(contrastive_loss.item())

        loss_total = lm_loss + contrastive_loss

        return loss_total, loss_dict

        

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def load_model(self):
        self.model = Steer(self.args, self.args.pretrain_name,  \
                                self.args.num_steers, self.args.rank, self.args.epsilon, self.args.init_var)
        self.tokenizer = self.model.tokenizer
        self.model.train()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        self.tokenizer.save_pretrained(path)

    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f'Training args {self.args}')

        batch_size = self.args.batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs
        
        optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)

        global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
        set_seed(self.args.seed)
        timer = Timer(total_steps)
        timer.start()
        self.model.train()

        loss_mean = RunningMean(self.args.gamma_mean)

        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)

                regularization_term = self.model.regularization_term()
                loss = loss + self.args.regularization * regularization_term

                loss /= self.args.grad_acc_steps

                loss.backward()
                acc_loss_dict.step(loss_dict)

                if (step+1) % self.args.grad_acc_steps == 0:
                    loss_mean.update(loss)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s, %s', idx+1, self.args.num_train_epochs, global_step, total_steps, acc_loss_pp, timer)
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_pp = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx+1, eval_loss_pp)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir)
                self.save(last_output_dir)

        if (idx+1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                eval_loss_pp = self.do_eval()
            self.args.logger.info('final eval loss: %s', eval_loss_pp)
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            self.args.logger.info('Saving model checkpoint to %s', last_output_dir)
            self.save(last_output_dir)
