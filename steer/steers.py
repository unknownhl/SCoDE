import torch
import torch.nn as nn

class Projected_Adaptor(nn.Module):
    def __init__(self, lm_head, num_steers, embed_dim, rank, epsilon, init_var, is_inference=False):
        super().__init__()
        assert rank > 0
        self.projector1 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
        self.projector2 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)

        self.rank = rank
        self.lm_head = lm_head
        self.epsilon = epsilon
        self.num_steers = num_steers
        self.init_var = init_var
        self.steer_values = torch.zeros(num_steers)
        self.is_inference = is_inference

        for name, param in self.named_parameters():
            if name not in ['projector1', 'projector2']:
                param.requires_grad = False

    def set_value(self, steer_values):
        device = self.projector1.device
        if not self.is_inference:
            self.steer_values = steer_values.to(device)
        else:
            self.steer_values_1 = steer_values[0].to(device)
            self.steer_values_2 = steer_values[1].to(device)

    def forward(self, state):
        if not self.is_inference:
            if self.steer_values.abs().sum() == 0:
                return state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
            delta = state[:, None].matmul(self.projector1[None]) *\
                    self.steer_values[:, :, None, None]
            delta = delta.matmul(
                    self.projector2.transpose(1, 2)[None]).sum(1)
            projected_state = state + self.epsilon * delta
            logits = projected_state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
            return logits
        else:
            if self.steer_values_1.abs().sum() == 0 and self.steer_values_2.abs().sum() == 0:
                return state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
            
            logits_ori = state.matmul(self.lm_head.weight.detach().transpose(0, 1))

            delta_neg = state[:, None].matmul(self.projector1[None]) *\
                        self.steer_values_1[:, :, None, None]
            delta_neg = delta_neg.matmul(
                        self.projector2.transpose(1, 2)[None]).sum(1)
            projected_state_neg = state + self.epsilon * delta_neg
            logits_neg = projected_state_neg.matmul(
                        self.lm_head.weight.detach().transpose(0, 1))
                
            delta_pos = state[:, None].matmul(self.projector1[None]) *\
                        self.steer_values_2[:, :, None, None]
            delta_pos = delta_pos.matmul(
                        self.projector2.transpose(1, 2)[None]).sum(1)
            projected_state_pos = state + self.epsilon * delta_pos
            logits_pos = projected_state_pos.matmul(
                        self.lm_head.weight.detach().transpose(0, 1))
                
            return logits_ori, logits_neg, logits_pos
               
    def regularization_term(self):
        return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)

