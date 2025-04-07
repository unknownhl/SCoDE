# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.generation.utils import GenerateDecoderOnlyOutput,GenerateEncoderDecoderOutput,GenerateBeamDecoderOnlyOutput,GenerateBeamEncoderDecoderOutput

from transformers.cache_utils import StaticCache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer

from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}

# # Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]



@torch.no_grad()
def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
       
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not self._supports_cache_class:
                raise ValueError(
                    "This model does not support the `cache_implementation` argument. Please check the following "
                    "issue: https://github.com/huggingface/transformers/issues/28981."
                )
            if generation_config.cache_implementation == "static":
                if not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs["past_key_values"] = self._get_static_cache(batch_size, generation_config.max_length)

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation == "static":
                raise ValueError("assisted generate is not supported with `static_cache`")

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. prepare logits warper (if `do_sample` is `True`)
            prepared_logits_warper = (
                self._get_logits_warper(generation_config) if generation_config.do_sample else None
            )

            # 13. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config) if generation_config.do_sample else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                is_inference=is_inference, ### modified
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config) if generation_config.do_sample else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        return result


@torch.no_grad()
def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList] = None,
        is_inference: bool = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            if not is_inference or not isinstance(outputs.logits, tuple): 
                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                if do_sample:
                    next_token_scores = logits_warper(input_ids, next_token_scores)
            
            # else:
            #     logits_ori, logits_neg, logits_pos = outputs.logits
            #     next_token_logits_ori = logits_ori[:, -1, :]
            #     next_token_logits_neg = logits_neg[:, -1, :]
            #     next_token_logits_pos = logits_pos[:, -1, :]

            #     next_token_scores_ori = logits_processor(input_ids, next_token_logits_ori)
            #     next_token_scores_neg = logits_processor(input_ids, next_token_logits_neg)
            #     next_token_scores_pos = logits_processor(input_ids, next_token_logits_pos)

            #     # Calculate C (confidence of the original model) and C_R (confidence of positive context)
            #     # C = torch.max(torch.nn.functional.softmax(next_token_scores_ori, dim=-1), dim=-1).values  # (batch_size)
            #     C_N = torch.max(torch.nn.functional.softmax(next_token_scores_neg, dim=-1), dim=-1).values  # (batch_size)
            #     C_P = torch.max(torch.nn.functional.softmax(next_token_scores_pos, dim=-1), dim=-1).values  # (batch_size)

            #     # Dynamic alpha calculation based on C and C_R

            #     # second best
            #     # dy_alpha = torch.where(C > C_P, 1 - C, C_P)  # alpha shape: (batch_size,)

            #     # best
            #     # dy_alpha = torch.where(C_N > C_P, 1 - C_N, C_P)  # alpha shape: (batch_size,)

            #     # Unsqueeze alpha to match the logits dimensions (batch_size, 1)
            #     # dy_alpha = dy_alpha.unsqueeze(-1)

            #     # second best
            #     # next_token_scores = next_token_scores_pos + dy_alpha * (next_token_scores_ori - next_token_scores_neg)

            #     # #TODO: JSD
            #     # # dy_alpha = get_jsd(next_token_scores_pos, next_token_scores_neg)
            #     # dy_alpha = get_jsd(next_token_scores_ori, next_token_scores_neg)

            #     # # test
            #     # next_token_scores =  next_token_scores_pos + dy_alpha * (next_token_scores_ori - next_token_scores_neg)

            #     alpha_1 = get_jsd(next_token_scores_ori, next_token_scores_neg)
            #     alpha_2 = get_jsd(next_token_scores_ori, next_token_scores_pos)
            #     next_token_scores = next_token_scores_ori - alpha_1 / (alpha_1 + alpha_2) * (next_token_scores_neg - next_token_scores_ori) + alpha_2 / (alpha_1 + alpha_2) * (next_token_scores_pos - next_token_scores_ori)

            #     # print('sample')
            #     # print(next_token_scores.shape)
            #     # print(next_token_scores)

            #     # best
            #     # next_token_scores = (1 + dy_alpha) * next_token_scores_pos - dy_alpha * next_token_scores_neg

            #     if do_sample:
            #         next_token_scores = logits_warper(input_ids, next_token_scores)
            
            # else:
            #     logits_ori, logits_neg, logits_pos = outputs.logits
            #     next_token_logits_ori = logits_ori[:, -1, :]
            #     next_token_logits_neg = logits_neg[:, -1, :]
            #     next_token_logits_pos = logits_pos[:, -1, :]

            #     print('next_token_logits_ori', next_token_logits_ori)
            #     # print('next_token_logits_neg', next_token_logits_neg)
            #     # print('next_token_logits_pos', next_token_logits_pos)

            #     # final_probs = relative_top_filter(next_token_logits_ori, 1)
            #     final_probs = relative_top_filter_probs(next_token_logits_ori, 0.1)
            #     print('final_probs', final_probs)
            #     # print('final_probs', final_probs)
            #     # probs_neg = next_token_logits_neg.log_softmax(dim=-1)
            #     # probs_pos = next_token_logits_pos.log_softmax(dim=-1)
            #     # mask = final_probs[0] < -1e3
            #     # probs_neg[0][mask] = -1e3
            #     # probs_pos[0][mask] = -1e3
            #     probs_neg = next_token_logits_neg.softmax(dim=-1)
            #     probs_pos = next_token_logits_pos.softmax(dim=-1)
            #     mask = final_probs[0] == 0.0
            #     print('mask', mask)
            #     # 判断mask多少个True
            #     print('mask', mask.sum())
            #     probs_neg[0][mask] = 0.0
            #     probs_pos[0][mask] = 0.0

            #     # print('probs_neg', probs_neg)
            #     # print('probs_pos', probs_pos)

            #     alpha_1 = get_jsd(next_token_logits_ori, next_token_logits_neg)
            #     alpha_2 = get_jsd(next_token_logits_ori, next_token_logits_pos)
            #     # alpha_3 = get_jsd(next_token_logits_neg, next_token_logits_pos)
            #     # print('ori-neg',alpha_1, 'ori-pos',alpha_2, 'pos-neg',alpha_3)

            #     next_token_logits = final_probs - alpha_1 / (alpha_1 + alpha_2) * (probs_neg - final_probs) + alpha_2 / (alpha_1 + alpha_2) * (probs_pos - final_probs)
            #     next_token_scores = next_token_logits

            else:
                logits_ori, logits_neg, logits_pos = outputs.logits
                next_token_logits_ori = logits_ori[:, -1, :]
                next_token_logits_neg = logits_neg[:, -1, :]
                next_token_logits_pos = logits_pos[:, -1, :]

                # final_logits = relative_top_filter(next_token_logits_ori, 0.1)
                # neg_logits = next_token_logits_neg.log_softmax(dim=-1)
                # pos_logits = next_token_logits_pos.log_softmax(dim=-1)
                # mask = final_logits[0] < -1e3
                # neg_logits[0][mask] = -1e3
                # pos_logits[0][mask] = -1e3
                
                # dynamic_alpha = get_jsd(next_token_logits_pos, next_token_logits_neg)

                # # multi-context
                # next_token_logits = final_logits + dynamic_alpha * (pos_logits - neg_logits)

                # pos_logits = relative_top_filter(next_token_logits_pos, 0.1)
                # neg_logits = next_token_logits_neg.log_softmax(dim=-1)
                # ori_logits = next_token_logits_ori.log_softmax(dim=-1)
                # mask = pos_logits[0] < -1e3
                # neg_logits[0][mask] = -1e3
                # ori_logits[0][mask] = -1e3
                
                # dynamic_alpha = get_jsd(pos_logits, neg_logits)

                # # multi-context
                # next_token_logits = ori_logits + dynamic_alpha * (pos_logits - neg_logits)
                
                # custom
                # alpha_1 = get_jsd(next_token_logits_ori, next_token_logits_neg)
                # alpha_2 = get_jsd(next_token_logits_ori, next_token_logits_pos)
                # pos_logits = relative_top_filter(next_token_logits_pos, 0.3)
                # mask = pos_logits[0] < -1e3
                # next_token_logits = next_token_logits_ori - alpha_1 / (alpha_1 + alpha_2) * (next_token_logits_neg - next_token_logits_ori) + alpha_2 / (alpha_1 + alpha_2) * (next_token_logits_pos - next_token_logits_ori)
                # next_token_logits = next_token_logits.log_softmax(dim=-1)
                # next_token_logits[0][mask] = -1e3

                # mu = calculate_mu(F.softmax(next_token_logits_pos, dim=-1))
                # high_uncertainty_indices = mu > 0.4e-2

                pos_logits = relative_top_filter(next_token_logits_pos, 0.3)
                neg_logits = next_token_logits_neg.log_softmax(dim=-1)
                ori_logits = next_token_logits_ori.log_softmax(dim=-1)
                mask = pos_logits[0] < -1e3
                neg_logits[0][mask] = -1e3
                ori_logits[0][mask] = -1e3
                alpha_1 = get_jsd(ori_logits, neg_logits)
                alpha_2 = get_jsd(ori_logits, pos_logits)

                # next_token_logits = ori_logits - alpha_1 / (alpha_1 + alpha_2) * (neg_logits - ori_logits) + alpha_2 / (alpha_1 + alpha_2) * (pos_logits - ori_logits)
                next_token_logits = ori_logits + alpha_1 / (alpha_1 + alpha_2) * (ori_logits - neg_logits) + alpha_2 / (alpha_1 + alpha_2) * (pos_logits - ori_logits)
                
                # next_token_logits[high_uncertainty_indices] = next_token_logits_pos.log_softmax(dim=-1)[high_uncertainty_indices]

                next_token_scores = logits_processor(input_ids, next_token_logits)
            
        
            # else:
            #     logits_ori, logits_neg, logits_pos = outputs.logits
            #     next_token_logits_ori = logits_ori[:, -1, :]
            #     next_token_logits_neg = logits_neg[:, -1, :]
            #     next_token_logits_pos = logits_pos[:, -1, :]

            #     threshold = 0.5e-2

            #     # Step 1: Calculate standard deviation to determine noise level
            #     mu = calculate_mu(next_token_logits_pos)

            #     # Step 2: Pre-judge the noise level and decide correctness
            #     y_bar = torch.mean(next_token_logits_pos, dim=1, keepdim=True)
            #     pos_updated_logits = torch.clone(next_token_logits_pos)
            #     neg_updated_logits = torch.clone(next_token_logits_neg)
            #     ori_updated_logits = torch.clone(next_token_logits_ori)


            #     high_uncertainty_mask = mu > threshold
            #     low_uncertainty_mask = ~high_uncertainty_mask

            #     # Apply selective contrastive decoding for low uncertainty cases
            #     pos_updated_logits[low_uncertainty_mask] = filter_logits(next_token_logits_pos[low_uncertainty_mask], y_bar[low_uncertainty_mask], 0.1)
            #     neg_updated_logits[low_uncertainty_mask] = filter_logits(next_token_logits_neg[low_uncertainty_mask], y_bar[low_uncertainty_mask], 0.1)
            #     ori_updated_logits[low_uncertainty_mask] = filter_logits(next_token_logits_ori[low_uncertainty_mask], y_bar[low_uncertainty_mask], 0.1)

            #     # Step 3: Calculate Jensen-Shannon Divergence or other metric for alpha weighting
            #     alpha_1 = get_jsd(ori_updated_logits, neg_updated_logits)
            #     alpha_2 = get_jsd(ori_updated_logits, pos_updated_logits)

            #     ori_updated_logits = torch.log_softmax(ori_updated_logits, dim=-1)
            #     neg_updated_logits = torch.log_softmax(neg_updated_logits, dim=-1)
            #     pos_updated_logits = torch.log_softmax(pos_updated_logits, dim=-1)

            #     # Step 4: Combine original and updated logits using dy_alpha
            #     next_token_logits = ori_updated_logits - alpha_1 / (alpha_1 + alpha_2) * (neg_updated_logits - ori_updated_logits) + alpha_2 / (alpha_1 + alpha_2) * (pos_updated_logits - ori_updated_logits)
                
            #     # # Step 1: Calculate standard deviation to determine noise level
            #     # mu = calculate_mu(next_token_logits_pos)

            #     # # Step 2: Pre-judge the noise level and decide correctness
            #     # y_bar = torch.mean(next_token_logits_pos, dim=1, keepdim=True)
            #     # pos_updated_logits = torch.clone(next_token_logits_pos)

            #     # for i in range(next_token_logits_pos.size(0)):
            #     #     if mu[i] > threshold:
            #     #         # High uncertainty, use original logits and mark as correct
            #     #         pass
            #     #     else:
            #     #         # Low uncertainty, apply selective contrastive decoding
            #     #         pos_updated_logits[i] = filter_logits(next_token_logits_pos[i], y_bar[i], 0.1)


            #     # dy_alpha = get_jsd(next_token_logits_ori, next_token_logits_pos)

            #     # next_token_logits = torch.where(pos_updated_logits != float('-inf'),
            #     #          torch.log_softmax(pos_updated_logits, dim=-1) - dy_alpha * torch.log_softmax(next_token_logits_ori, dim=-1),
            #     #          torch.tensor(float('-inf')))

            #     next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def calculate_mu(logits):
    y_bar = torch.mean(logits, dim=-1, keepdim=True)
    mu = torch.sqrt(torch.mean((logits - y_bar) ** 2, dim=-1))
    return mu

def filter_logits(logits, y_bar, eta):
    threshold = eta * y_bar
    filtered_logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf')))
    return filtered_logits


def relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value

    # 避免 scores_normalized 中的值出现 inf 或 nan
    scores_normalized = scores_normalized.clamp(min=-1e9, max=1e9)

    return scores_normalized


def relative_top_filter_(scores: torch.FloatTensor, relative_top: float = 0.1) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    y_bar = torch.mean(scores_normalized, dim=1, keepdim=True) 
    threshold = relative_top * y_bar
    scores_normalized = torch.where(scores_normalized >= threshold, scores_normalized, torch.tensor(float('-inf')))
    # 避免 scores_normalized 中的值出现 inf 或 nan
    scores_normalized = scores_normalized.clamp(min=-1e9, max=1e9)
    return scores_normalized

# def get_jsd(p, q):
#     p = F.softmax(p, dim=-1)
#     q = F.softmax(q, dim=-1)
#     p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
#     if ((p + q) == 0).any():
#         m = (0.5 * (p + q)).clamp_min(1e-9).log()
#     else:
#         m = (0.5 * (p + q)).log()
#     if torch.any(p <= 0):
#         p = p.clamp_min(1e-9)
#     if torch.any(q <= 0):
#         q = q.clamp_min(1e-9)
#     return 0.5 * (F.kl_div(m, p, reduction='batchmean', log_target=False) + F.kl_div(m, q, reduction='batchmean', log_target=False))


# def get_jsd(p, q):
#     p = F.softmax(p, dim=-1)
#     q = F.softmax(q, dim=-1)
    
#     # Clamping to avoid zeros
#     p = torch.where(p > 0, p, torch.tensor(1e-9, device=p.device))
#     q = torch.where(q > 0, q, torch.tensor(1e-9, device=q.device))
    
#     m = (0.5 * (p + q)).clamp(min=1e-9).log()
    
#     return 0.5 * (F.kl_div(m, p, reduction='batchmean', log_target=False) + 
#                   F.kl_div(m, q, reduction='batchmean', log_target=False))


def get_jsd(p, q, epsilon=1e-9):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)
    m = 0.5 * (p + q)
    jsd = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + 
                 F.kl_div(q.log(), m, reduction='batchmean'))
    return jsd
