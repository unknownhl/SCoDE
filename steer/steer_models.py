import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers.generation.utils as gu
import transformers.models.llama.modeling_llama as ll
import transformers.models.qwen2.modeling_qwen2 as qw
import transformers.models.codegen.modeling_codegen as cg
import transformers.models.mistral.modeling_mistral as mi

from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS
from steer.steers import Projected_Adaptor
from steer.steer_utils import Hack_no_grad
from steer._sample import generate, sample, text_sample
from steer._codellama import custom_LlamaForCausalLM
from steer._qwen2 import custom_Qwen2ForCausalLM
from steer._codegen import custom_CodeGenForCausalLM
from steer._mistral import custom_MistralForCausalLM

class Steer(nn.Module):
    def __init__(self, args, model_name, num_steers, rank, epsilon, init_var, **kwargs):
        super().__init__()

        self.args = args
        self.device = torch.device(args.device)
        self.is_inference = args.is_inference

        # model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')
        model_dir = self._get_model_directory(model_name)
        print(f"Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if 'qwen2.5-coder-1b' == model_name.lower() or 'qwen2.5-coder-3b' == model_name.lower() or 'qwen2.5-coder-1b-steer' == model_name.lower() or 'qwen2.5-coder-3b-steer' == model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map=self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, device_map=self.device, vocab_size=len(self.tokenizer)
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.config = self.model.config
        self.generation_config = GenerationConfig.from_pretrained(model_dir)

        self.init_var = init_var
        self.num_steers = num_steers
        self.embed_dim = self.model.lm_head.weight.shape[1]

        for _param in self.model.parameters():
            _param.requires_grad_(False)

        if 'starcoder' in model_name or 'codegen' in model_name:
            self.model.transformer = Hack_no_grad(self.model.transformer)
        if 'codellama' in model_name or 'deepseek' in model_name:
            self.model.model = Hack_no_grad(self.model.model)
        if 'qwen2' in model_name:
            self.model.model = Hack_no_grad(self.model.model)

        self.model.lm_head = Projected_Adaptor(
            self.model.lm_head, num_steers, self.embed_dim, rank, epsilon, init_var, is_inference=self.is_inference, **kwargs
        )
        self.model.to(self.device)
    
    def get_embed_dim(self):
        return self.embed_dim

    def forward(self, input_ids, steer_values, **kwargs):
        """Forward pass through the model with steer values."""
        self.model.lm_head.set_value(steer_values)
        output = self.model(input_ids, **kwargs)
        return output

    def to_device(self, device):
        """Move model to specified device."""
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        """Return regularization term for the projected adaptor."""
        return self.model.lm_head.regularization_term()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, input_ids, steer_values, **kwargs):
        """Generate text using the model with steer values."""
        steer_values_tensor = self._prepare_steer_values(steer_values)
        self.model.lm_head.set_value(steer_values_tensor)

        if self.is_inference:
            # Patch the generation functions for custom sampling
            ll.LlamaForCausalLM.forward = custom_LlamaForCausalLM.forward
            qw.Qwen2ForCausalLM.forward = custom_Qwen2ForCausalLM.forward
            cg.CodeGenForCausalLM.forward = custom_CodeGenForCausalLM.forward
            mi.MistralForCausalLM.forward = custom_MistralForCausalLM.forward
            gu.GenerationMixin.generate = generate
            gu.GenerationMixin._sample = sample

            # Generate output
            gen_output = self.model.generate(
                input_ids, 
                do_sample = kwargs["do_sample"], 
                num_return_sequences = kwargs["num_return_sequences"], 
                temperature = kwargs["temperature"], 
                max_new_tokens = kwargs["max_new_tokens"],
                top_p = kwargs["top_p"],
                pad_token_id = kwargs["pad_token_id"],
                eos_token_id = kwargs["eos_token_id"],
                use_cache = kwargs["use_cache"],
                is_inference = self.is_inference,
            )
        else:
            gen_output = self.model.generate(
                input_ids, 
                do_sample = kwargs["do_sample"], 
                num_return_sequences = kwargs["num_return_sequences"], 
                temperature = kwargs["temperature"], 
                max_new_tokens = kwargs["max_new_tokens"],
                top_p = kwargs["top_p"],
                pad_token_id = kwargs["pad_token_id"],
                eos_token_id = kwargs["eos_token_id"],
                use_cache = kwargs["use_cache"],
            )
        return gen_output

    
    def _prepare_steer_values(self, steer_values):
        """Helper function to prepare steer values tensor."""
        if self.is_inference:
            # steer_values_1 = torch.tensor([steer_values[0], 0], device=self.device)[None]
            # steer_values_2 = torch.tensor([0, steer_values[-1]], device=self.device)[None]
            steer_values_1 = torch.tensor([steer_values[0], 1], device=self.device)[None]
            steer_values_2 = torch.tensor([steer_values[-1], 1], device=self.device)[None]
            return [steer_values_1, steer_values_2]
        else:
            return torch.tensor(steer_values, device=self.device)[None]
        
    def _get_model_directory(self, model_name):
        """Helper function to get model directory based on model name."""
        if model_name in PRETRAINED_MODELS:
            return PRETRAINED_MODELS[model_name]
        elif model_name in CHAT_MODELS:
            return CHAT_MODELS[model_name]
        elif "std" in model_name:
            print(model_name.split('-steer')[0])
            return os.path.join(self.args.model_dir, model_name.split('-steer')[0], 'checkpoint-last')
        elif "steer" in model_name:
            print("**ablate for function tuning**")
            print(model_name)
            return PRETRAINED_MODELS[model_name.split('-steer')[0]]
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        

