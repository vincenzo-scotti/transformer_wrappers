import os
import logging
from datetime import datetime
import inspect
from copy import deepcopy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from torchmetrics import MetricCollection

from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaPreTrainedModel, GPT2PreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel, Gemma2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
)
from transformers import logging as hf_logging

from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModel, AutoPeftModelForCausalLM
from peft.peft_model import PeftModel

from .constants import *
from .base import AttentionWrapper, LayerWrapper, LayersWrapper, TransformerWrapper, CausalLMWrapper
from transformer_wrappers.optim import optimizer_mapping, lr_scheduler_mapping
from transformer_wrappers.utils.metrics import PPLScore

from enum import Enum
from typing import Union, Optional, Type, Tuple, Dict, List, Iterable

# TODO fix base model/module properties and wrapper enable/disable methods
# TODO implement gradient checkpointing
# TODO implement training with adapters
# TODO test past key-value


ATT_LEN: str = 'att_len'


__all__ = [
    'SHARED_STRUCTURE_MODELS',
    'SHARED_STRUCTURE_LAYERS',
    'ModuleWrapper',
    'PreTrainedModelWrapper',
    'EmbeddingWrapper',
    'AttentionWrapper',
    'FeedForwardWrapper',
    'LayerWrapper',
    'LayersWrapper',
    'LMHeadWrapper',
    'TransformerWrapper',
    'CausalLMWrapper'
]


logger = hf_logging.get_logger(__name__)


SHARED_STRUCTURE_MODELS = (GemmaPreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel, Gemma2PreTrainedModel)
SHARED_STRUCTURE_LAYERS = (GemmaDecoderLayer, LlamaDecoderLayer, MistralDecoderLayer, Gemma2DecoderLayer)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ContextAttentionWrapper(AttentionWrapper):
    def _wrapped_forward(
            self,
            current_hidden_state: Optional[torch.FloatTensor] = None,
            attention_params: Optional[Dict] = None,
            att_len: Optional[int] = None,
            **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()
        if attention_params is None:
            raise ValueError()
        #
        fine_grained_output = True
        #
        if not fine_grained_output:
            attn_output = self.base_module.forward(current_hidden_state, **attention_params)
        elif isinstance(self.super_wrapper.base_module, GPT2Block):
            bsz, q_len, _ = current_hidden_state.size()
            attention_mask = attention_params.get('attention_mask', None)
            head_mask = attention_params.get('head_mask', None)
            encoder_hidden_states = attention_params.get('encoder_hidden_states', None)
            layer_past = attention_params.get('layer_past', None)
            use_cache = attention_params.get('use_cache', None)
            encoder_attention_mask = attention_params.get('encoder_attention_mask', None)
            output_attentions = attention_params.get('output_attentions', None)

            
            if encoder_hidden_states is not None:
                if not hasattr(self, "q_attn"):
                    raise ValueError(
                        "If class is used as cross attention, the weights `q_attn` have to be defined. "
                        "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                    )

                query = self.base_module.q_attn(current_hidden_state)
                key, value = self.base_module.c_attn(encoder_hidden_states).split(self.base_module.split_size, dim=2)
                attention_mask = encoder_attention_mask
            else:
                query, key, value = self.base_module.c_attn(current_hidden_state).split(self.base_module.split_size, dim=2)

            query = self.base_module._split_heads(query, self.base_module.num_heads, self.base_module.head_dim)
            key = self.base_module._split_heads(key, self.base_module.num_heads, self.base_module.head_dim)
            value = self.base_module._split_heads(value, self.base_module.num_heads, self.base_module.head_dim)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            if self.base_module.reorder_and_upcast_attn:
                attn_output, attn_weights = self.base_module._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
            else:
                # New
                # reimplementing only this attention forward since reorder_and_upcast_attn is usually false  (gdm), 
                                
                attn_weights = torch.matmul(query, key.transpose(-1, -2))

                if self.base_module.scale_attn_weights:
                    attn_weights = attn_weights / torch.full(
                        [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                    )

                # Layer-wise attention scaling
                if self.base_module.scale_attn_by_inverse_layer_idx:
                    attn_weights = attn_weights / float(self.base_module.layer_idx + 1)

                if not self.base_module.is_cross_attention:
                    # if only "normal" attention layer implements causal mask
                    query_length, key_length = query.size(-2), key.size(-2)
                    causal_mask = self.base_module.bias[:, :, key_length - query_length : key_length, :key_length]
                    mask_value = torch.finfo(attn_weights.dtype).min
                    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
                    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

                if attention_mask is not None:
                    # Apply the attention mask
                    # NOTE: New
                    # Apply length mask
                    if att_len is not None:
                        coords = torch.arange(q_len * key_length, device=attention_mask.device).reshape(q_len, key_length)
                        mask = (coords % key_length) + att_len <= (coords // key_length) + (key_length - q_len)
                        # TODO 
                        attention_mask = attention_mask.expand((bsz, self.base_module.num_heads, q_len, q_len))
                        attention_mask[..., mask] = torch.finfo(current_hidden_state.dtype).min
                    attn_weights = attn_weights + attention_mask
                elif  att_len is not None and (attention_mask is None or attention_mask.numel()==1):
                    # During generation
                    attn_weights[..., :-att_len] = torch.finfo(current_hidden_state.dtype).min
                else:
                    raise ValueError("Wrong attention_maks, bitch!")

                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
                attn_weights = attn_weights.type(value.dtype)
                attn_weights = self.base_module.attn_dropout(attn_weights)

                # Mask heads if we want to
                if head_mask is not None:
                    attn_weights = attn_weights * head_mask

                attn_output = torch.matmul(attn_weights, value)

            attn_output = self.base_module._merge_heads(attn_output, self.base_module.num_heads, self.base_module.head_dim)
            attn_output = self.base_module.c_proj(attn_output)
            attn_output = self.base_module.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            
            attn_output = outputs

        elif isinstance(self.super_wrapper.base_module, GPTNeoXLayer):
            
            bsz, q_len, _ = current_hidden_state.size()
            head_mask = attention_params.get('head_mask', None)
            layer_past = attention_params.get('layer_past', None)
            use_cache = attention_params.get('use_cache', None)
            position_ids = attention_params.get('position_ids', None)
            attention_mask = attention_params.get('attention_mask', None)
            output_attentions = attention_params.get('output_attentions', None)

            # Apply attention-specific projections and rope
            query, key, value, present = self.base_module._attn_projections_and_rope(
                hidden_states=current_hidden_state, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
            )

            # Compute attention
            # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
            # compute causal mask from causal mask buffer
            batch_size, num_attention_heads, query_length, attn_head_size = query.size()
            key_length = key.size(-2)

            # dynamically increase the causal mask with the key length, if needed.
            if key_length > self.base_module.bias.shape[-1]:
                self.base_module._init_bias(key_length, device=key.device)
            causal_mask = self.base_module.bias[:, :, key_length - query_length : key_length, :key_length]

            query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
            key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
            attn_scores = torch.zeros(
                batch_size * num_attention_heads,
                query_length,
                key_length,
                dtype=query.dtype,
                device=key.device,
            )
            attn_scores = torch.baddbmm(
                attn_scores,
                query,
                key.transpose(1, 2),
                beta=1.0,
                alpha=self.base_module.norm_factor,
            )
            attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

            mask_value = torch.finfo(attn_scores.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
            attn_scores = torch.where(causal_mask, attn_scores, mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                # NOTE: New
                # Apply length mask
                # Attention mask is passed as a slice of a preallocated attention mask?
                # Every modification is directly mapped to the original attention_mask?
                # No! remembber that there during generation there is another forward pass!
                if att_len is not None:
                    coords = torch.arange(q_len * key_length, device=attention_mask.device).reshape(q_len, key_length)
                    mask = (coords % key_length) + att_len <= (coords // key_length) + (key_length - q_len)
                    attention_mask[..., mask] = torch.finfo(current_hidden_state.dtype).min

                attn_scores = attn_scores + attention_mask

            attn_weights = nn.functional.softmax(attn_scores, dim=-1)
            attn_weights = attn_weights.to(value.dtype)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_weights = self.base_module.attention_dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, value)
        

            # Reshape outputs
            attn_output = self.base_module._merge_heads(attn_output, self.base_module.num_attention_heads, self.base_module.head_size)
            attn_output = self.base_module.dense(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            
            attn_output = outputs

        elif isinstance(self.super_wrapper.base_module, SHARED_STRUCTURE_LAYERS):
            position_ids = attention_params.get("position_ids", None)
            past_key_value = attention_params.get("past_key_value", None)
            cache_position = kwargs.get("cache_position", None)
            attention_mask = attention_params.get("attention_mask", None)
            output_attentions = kwargs.get("output_attentions", None)

            # Force init of attention mask and check its length, if it is lower than 
            # the generated output so far append another row and column
            #if self.attention_mask is None and attention_mask is not None:
            #    self.attention_mask = attention_mask
            #if self.attention_mask.shape[-1] < q_len + int(cache_position.squeeze()):
            #    # slice last row
            #    last_row = self.attention_mask[:, :, -1, :]
            #    # slice last column
            #    last_column = self.attention_mask[:, :, :, -1]
            #    # Append to last row and last column the 0 value of the diagonal
            #    #torch.cat

            
            bsz, q_len, _ = current_hidden_state.size()
            query_states = self.base_module.q_proj(current_hidden_state)
            key_states = self.base_module.k_proj(current_hidden_state)
            value_states = self.base_module.v_proj(current_hidden_state)

            query_states = query_states.view(bsz, q_len, self.base_module.num_heads, self.base_module.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.base_module.num_key_value_heads, self.base_module.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.base_module.num_key_value_heads, self.base_module.head_dim).transpose(1, 2)

            cos, sin = self.base_module.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.base_module.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.base_module.num_key_value_groups)
            value_states = repeat_kv(value_states, self.base_module.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.base_module.head_dim)
            
            kv_seq_len = key_states.shape[-2]
            if attention_mask is not None and attention_mask.numel()!=1:  # no matter the length, we just slice it
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                # NOTE: New
                # Apply length mask
                # Attention mask is passed as a slice of a preallocated attention mask?
                # Every modification is directly mapped to the original attention_mask?
                if att_len is not None:
                    coords = torch.arange(q_len * kv_seq_len, device=attention_mask.device).reshape(q_len, kv_seq_len)
                    mask = (coords % kv_seq_len) + att_len <= (coords // kv_seq_len) + (kv_seq_len - q_len)
                    attention_mask[..., mask] = torch.finfo(current_hidden_state.dtype).min

                attn_weights = attn_weights + attention_mask
            elif  att_len is not None and (attention_mask is None or attention_mask.numel()==1):
                # During generation
                attn_weights[..., :-att_len] = torch.finfo(current_hidden_state.dtype).min
            else:
                raise ValueError("Wrong attention_maks, bitch!")

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.base_module.attention_dropout, training=self.base_module.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.base_module.num_heads, q_len, self.base_module.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.base_module.num_heads, q_len, self.base_module.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.base_module.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            attn_output =  (attn_output, attn_weights, past_key_value)
        else:
            raise NotImplementedError(f'Unsupported layer type: `{type(self.super_wrapper.base_module)}`.')
        #
        output = kwargs | {self.module_output: attn_output, CURR_HIDDEN_STATE: current_hidden_state, ATT_LEN: att_len}

        return output


class ContextLayerWrapper(LayerWrapper):
    _attention_dtype: Type[AttentionWrapper] = ContextAttentionWrapper


class ContextLayersWrapper(LayersWrapper):

    _layer_dtype: Type[LayerWrapper] = ContextLayerWrapper


class ContextTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[LayersWrapper] = ContextLayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # To declare a new attribute for the transformer
        self.att_len: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(ATT_LEN)

    def _pre_process_input(self, *args, att_len: Optional[int] = None, **kwargs):
        kwargs = super()._pre_process_input(*args, **kwargs)
        #
        att_len = att_len if att_len is not None else self.att_len
        #
        kwargs |= {
            ATT_LEN: att_len
        }

        return kwargs
    
    
class ContextCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = ContextTransformerWrapper

    def prepare_inputs_for_generation(
            self,
            *args,
            att_len: Optional[int] = None,
            **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs) | {
            ATT_LEN: att_len
        }

        return inputs
