import os
import inspect

from enum import Enum
from typing import Union, Optional, Type, Tuple, Dict, List, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaPreTrainedModel, GPT2PreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
)
from transformers import logging

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft.peft_model import PeftModel

from .constants import *

# TODO fix base model/module properties and wrapper enable/disable methods
# TODO implement gradient checkpointing
# TODO implement training with adapters
# TODO test past key-value


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


logger = logging.get_logger(__name__)


SHARED_STRUCTURE_MODELS = (GemmaPreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel)
SHARED_STRUCTURE_LAYERS = (GemmaDecoderLayer, LlamaDecoderLayer, MistralDecoderLayer)


class TransformerEmbeddingAttr(Enum):
    EMBED_TOKENS = 'embed_tokens'
    WTE = 'wte'


class TransformerPositionEmbeddingAttr(Enum):
    WPE = 'wpe'


class LayerInitialNormAttr(Enum):
    INPUT_LAYERNORM = 'input_layernorm'
    LN_1 = 'ln_1'


class LayerAttentionAttr(Enum):
    SELF_ATTN = 'self_attn'
    ATTN = 'attn'


class LayerIntermediateNormAttr(Enum):
    INPUT_LAYERNORM = 'post_attention_layernorm'
    LN_2 = 'ln_2'


class LayerFeedForwardAttr(Enum):
    MLP = 'mlp'


class TransformerLayersAttr(Enum):
    LAYERS = 'layers'
    H = 'h'


class TransformerNormAttr(Enum):
    NORM = 'norm'
    LN_F = 'ln_f'


class LMTransformerAttr(Enum):
    MODEL = 'model'
    TRANSFORMER = 'transformer'
    
    
class LMHeadAttr(Enum):
    LM_HEAD = 'lm_head'


AttrEnumTypes: Type = Union[
    LayerInitialNormAttr, LayerAttentionAttr, LayerIntermediateNormAttr, LayerFeedForwardAttr,
    TransformerEmbeddingAttr, TransformerPositionEmbeddingAttr, TransformerLayersAttr, TransformerNormAttr,
    LMTransformerAttr, LMHeadAttr
]


def _get_module_attr_name(model: nn.Module, attr_names: Type[AttrEnumTypes]):
    #
    for attr in attr_names:
        if hasattr(model, attr.value):
            return attr
    #
    raise ValueError(f'Unsupported module type `{type(model)}` for attribute `{attr_names}`.')


class BaseWrapper:
    @property
    def is_wrapping(self) -> bool:
        return True

    def enable_wrapper(self):
        pass

    def disable_wrapper(self):
        pass

    def _pre_process_input(self, *args, **kwargs):
        return kwargs

    def _wrapped_forward(self, **kwargs):
        raise NotImplementedError()

    def _post_process_output(self, **kwargs):
        return kwargs


class ModuleWrapper(nn.Module, BaseWrapper):
    _module_name: str = 'module'
    module_output: str = 'module_output'
    module_attr: Optional[Type[AttrEnumTypes]] = None

    def __init__(self, module: nn.Module, super_wrapper: Optional[BaseWrapper] = None):
        super().__init__()
        self._module: nn.Module = module
        self._super_wrapper: Optional[BaseWrapper] = super_wrapper

    @property
    def base_module(self) -> nn.Module:
        # logger.warning(
        #     f'The returned base {self._module_name} may be modified and may have internal wrappers still enabled.'
        # )
        return self._module

    @property
    def super_wrapper(self) -> BaseWrapper:
        return self._super_wrapper

    def eval(self):
        self._module = self._module.eval()

        return self

    def train(self, mode: bool = True):
        self._module = self._module.train(mode=mode)

        return self

    def _wrapped_forward(self, **kwargs):
        # Model forward
        module_output = self.base_module.forward(**kwargs)
        # Extend input with module output
        output = kwargs | {self.module_output: module_output}

        return output

    def _post_process_output(self, base_model_output: bool = False, **kwargs):
        if base_model_output:
            return kwargs[self.module_output]
        else:
            return kwargs

    def forward(self, *args, base_model_output: bool = False, **kwargs):
        self.enable_wrapper()
        # Pre-process input
        kwargs = self._pre_process_input(*args, **kwargs)
        # Apply layer transformation
        output = self._wrapped_forward(**kwargs)
        # Post-process output
        output = self._post_process_output(base_model_output=base_model_output, **output)

        return output


class EmbeddingWrapper(ModuleWrapper):
    _module_name: str = 'embedding module'
    module_output: str = 'embedding_output'
    module_attr = TransformerEmbeddingAttr

    def __init__(self, *args, position_embeddings: Optional[nn.Embedding] = None, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self._position_embeddings: Optional[nn.Embedding] = position_embeddings

    def _wrapped_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_embeddings: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        #
        input_embeddings = input_embeddings if input_embeddings is not None else self.base_module.forward(input_ids)
        #
        output = kwargs | {self.module_output: input_embeddings}

        return output

    def _post_process_output(self, base_model_output: bool = False, **kwargs):
        embeddings = kwargs.pop(self.module_output)
        if isinstance(self.super_wrapper.base_model, GPT2PreTrainedModel):
            embeddings += self._position_embeddings.forward(kwargs[POSITION_IDS])
        elif isinstance(self.super_wrapper.base_model, GemmaPreTrainedModel):
            embeddings *= embeddings.size(-1) ** 0.5
        if base_model_output:
            return embeddings
        else:
            kwargs |= {EMBEDDINGS: embeddings}

            return kwargs


class AttentionWrapper(ModuleWrapper):
    _module_name: str = 'attention module'
    module_output: str = ATTN_OUTPUT
    module_attr = LayerAttentionAttr

    attention_weights: str = CURR_ATTN_WEIGHTS
    key_value: str = CURR_KEY_VALUE

    def _pre_process_input(self, *args, layer_idx: Optional[int] = None, **kwargs):
        #
        attention_params = {
            ATTENTION_MASK: kwargs[ATTENTION_MASK],
            OUTPUT_ATTENTIONS: kwargs[OUTPUT_ATTENTIONS],
            USE_CACHE: kwargs[USE_CACHE]
        }
        if isinstance(self.super_wrapper.super_wrapper.super_wrapper.base_model, GPT2PreTrainedModel):
            attention_params |= {LAYER_PAST: kwargs[PAST_KEY_VALUES][layer_idx]}
        elif isinstance(self.super_wrapper.super_wrapper.super_wrapper.base_model, SHARED_STRUCTURE_MODELS):
            attention_params |= {PAST_KEY_VALUE: kwargs[PAST_KEY_VALUES]}
        else:
            raise NotImplementedError(
                f'Unsupported model type: `{type(self.super_wrapper.super_wrapper.super_wrapper.base_model)}`.'
            )
        if isinstance(self.super_wrapper.super_wrapper.super_wrapper.base_model, SHARED_STRUCTURE_MODELS):
            attention_params |= {POSITION_IDS: kwargs[POSITION_IDS]}
        #
        kwargs |= {ATTN_PARAMS: attention_params}

        return kwargs

    def _wrapped_forward(
            self,
            current_hidden_state: Optional[torch.FloatTensor] = None,
            attention_params: Optional[Dict] = None,
            **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()
        if attention_params is None:
            raise ValueError()
        #
        attn_output = self.base_module.forward(current_hidden_state, **attention_params)
        #
        output = kwargs | {self.module_output: attn_output, CURR_HIDDEN_STATE: current_hidden_state}

        return output

    def _post_process_output(
            self,
            base_model_output: bool = False,
            output_attentions: bool = False,  # Output attention weights
            **kwargs
    ):
        #
        if base_model_output:
            #
            return kwargs[self.module_output]
        else:
            #
            module_output = kwargs.pop(self.module_output)
            if len(module_output) == 3:
                if isinstance(self.super_wrapper.base_module, SHARED_STRUCTURE_LAYERS):
                    output = dict(zip((self.module_output, self.attention_weights, self.key_value), module_output))
                else:
                    output = dict(zip((self.module_output, self.key_value, self.attention_weights), module_output))
            elif len(module_output) == 2:
                if output_attentions:
                    output = dict(zip((self.module_output, self.attention_weights), module_output))
                else:
                    output = dict(zip((self.module_output, self.key_value), module_output))
            elif len(module_output) == 1:
                output = {dict(zip((self.module_output,), module_output))}
            else:
                raise ValueError()
            #
            kwargs |= {self.module_output: output, OUTPUT_ATTENTIONS: output_attentions}

            return kwargs


class FeedForwardWrapper(ModuleWrapper):
    _module_name: str = 'feed forward module'
    module_output: str = FFNN_OUTPUT
    module_attr = LayerFeedForwardAttr

    def _wrapped_forward(self, current_hidden_state: Optional[torch.FloatTensor], **kwargs):
        if current_hidden_state is None:
            raise ValueError()
        #
        ffnn_output = self.base_module.forward(current_hidden_state)
        #
        output = kwargs | {self.module_output: ffnn_output, CURR_HIDDEN_STATE: current_hidden_state}

        return output


class LayerWrapper(ModuleWrapper):
    _module_name: str = 'layer module'
    module_output: str = 'layer_output'

    _attention_dtype: Type[ModuleWrapper] = AttentionWrapper
    _feed_forward_dtype: Type[ModuleWrapper] = FeedForwardWrapper

    def __init__(self, layer: nn.Module, *args, **kwargs):
        super().__init__(layer, *args, **kwargs)
        # Attribute names
        self._initial_norm_attr: LayerInitialNormAttr = self._get_initial_norm_attr()
        self._attention_attr: LayerAttentionAttr = self._get_attention_attr()
        self._intermediate_norm_attr: LayerIntermediateNormAttr = self._get_intermediate_norm_attr()
        self._feed_forward_attr: LayerFeedForwardAttr = self._get_feed_forward_attr()
        # Wrappers
        self._attention_wrapper = self._attention_dtype(getattr(self._module, self._attention_attr.value), super_wrapper=self)
        self._feed_forward_wrapper = self._feed_forward_dtype(getattr(self._module, self._feed_forward_attr.value), super_wrapper=self)

    @property
    def is_attention_wrapping(self):
        return isinstance(getattr(self.base_module, self._attention_attr.value), self._attention_dtype)

    @property
    def is_feed_forward_wrapping(self):
        return isinstance(getattr(self.base_module, self._feed_forward_attr.value), self._feed_forward_dtype)

    @property
    def is_wrapping(self):
        return self.is_attention_wrapping or self.is_feed_forward_wrapping  # TODO Decide for 'or' or 'and'

    def enable_attention_wrapper(self):
        if not self.is_attention_wrapping:
            setattr(self.base_module, self._attention_attr.value, self.attention_wrapper)

    def enable_feed_forward_wrapper(self):
        if not self.is_feed_forward_wrapping:
            setattr(self.base_module, self._feed_forward_attr.value, self.feed_forward_wrapper)

    def enable_wrapper(self):
        if not self.is_wrapping:
            self.enable_attention_wrapper()
            self.enable_feed_forward_wrapper()

    def disable_attention_wrapper(self):
        if self.is_attention_wrapping:
            setattr(self.base_module, self._attention_attr.value, self.attention_wrapper.base_module)

    def disable_feed_forward_wrapper(self):
        if self.is_feed_forward_wrapping:
            setattr(self.base_module, self._feed_forward_attr.value, self.feed_forward_wrapper.base_module)

    def disable_wrapper(self):
        if self.is_wrapping:
            self.disable_attention_wrapper()
            self.disable_feed_forward_wrapper()

    def _get_initial_norm_attr(self) -> LayerInitialNormAttr:
        return _get_module_attr_name(self.base_module, LayerInitialNormAttr)

    def _get_attention_attr(self) -> LayerAttentionAttr:
        return _get_module_attr_name(self.base_module, LayerAttentionAttr)

    def _get_intermediate_norm_attr(self) -> LayerIntermediateNormAttr:
        return _get_module_attr_name(self.base_module, LayerIntermediateNormAttr)

    def _get_feed_forward_attr(self) -> LayerFeedForwardAttr:
        return _get_module_attr_name(self.base_module, LayerFeedForwardAttr)

    @property
    def initial_norm(self) -> nn.Module:
        return getattr(self.base_module, self._initial_norm_attr.value)

    @property
    def attention(self):
        return self.attention_wrapper.base_module

    @property
    def attention_wrapper(self) -> AttentionWrapper:
        return self._attention_wrapper

    @property
    def intermediate_norm(self) -> nn.Module:
        return getattr(self.base_module, self._intermediate_norm_attr.value)

    @property
    def feed_forward(self):
        return self.feed_forward_wrapper.base_module

    @property
    def feed_forward_wrapper(self):
        return self._feed_forward_wrapper

    def _attn_forward(
            self, current_hidden_state: Optional[torch.FloatTensor], add_attn_residual: bool = True, **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Initial Normalisation
        current_hidden_state = self.initial_norm.forward(current_hidden_state)
        # Self attention
        attention_output = self.attention_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.attention_wrapper.module_output)
        if add_attn_residual:
            current_hidden_state = attention_output[self.attention_wrapper.module_output] + residual
        else:
            current_hidden_state = attention_output[self.attention_wrapper.module_output]
        #
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_ATTN_RESIDUAL: add_attn_residual,
            self.attention_wrapper.module_output: attention_output
        }

        return output

    def _ffnn_forward(self, current_hidden_state, add_ffnn_residual: bool = True, **kwargs):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Intermediate Normalisation
        current_hidden_state = self.intermediate_norm.forward(current_hidden_state)
        # Feed-Forward
        ffnn_output = self.feed_forward_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.feed_forward_wrapper.module_output)
        if add_ffnn_residual:
            current_hidden_state = ffnn_output + residual  # TODO verify this
        else:
            current_hidden_state = ffnn_output
        # Extend input with module output
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_FFNN_RESIDUAL: add_ffnn_residual,
            self.feed_forward_wrapper.module_output: ffnn_output
        }

        return output

    def _wrapped_forward(self, skip_attention: bool = False, skip_ffnn: bool = False, **kwargs):
        output = kwargs
        output = self._attn_forward(**output)
        output = self._ffnn_forward(**output)
        #
        output |= {self.module_output: output[CURR_HIDDEN_STATE]}

        return output

    def _post_process_output(
            self,
            base_model_output: bool = False,
            use_cache: bool = False,
            output_attentions: bool = False,  # Output attention weights
            return_attention_output: bool = False,  # Self-attention layer output
            return_feed_forward_output: bool = False,
            **kwargs
    ):
        layer_output = kwargs.pop(self.module_output)
        attention_output = kwargs.pop(self.attention_wrapper.module_output)
        feed_forward_output = kwargs.pop(self.feed_forward_wrapper.module_output)
        if base_model_output:
            if output_attentions and use_cache:
                return (
                    layer_output,
                    attention_output[self.attention_wrapper.attention_weights],
                    attention_output[self.attention_wrapper.key_value]
                )
            elif output_attentions:
                return layer_output, attention_output[self.attention_wrapper.attention_weights]
            elif use_cache:
                return layer_output, attention_output[self.attention_wrapper.key_value]
            else:
                return layer_output,
        else:
            output = {CURR_HIDDEN_STATE: layer_output}
            if output_attentions:
                output[CURR_ATTN_WEIGHTS] = attention_output[self.attention_wrapper.attention_weights]
            if use_cache:
                output[CURR_KEY_VALUE] = attention_output[self.attention_wrapper.key_value]
            if return_attention_output:
                output[ATTN_OUTPUT] = attention_output[self.attention_wrapper.module_output]
            if return_feed_forward_output:
                output[FFNN_OUTPUT] = feed_forward_output

            kwargs |= {
                self.module_output: output,
                USE_CACHE: use_cache,
                OUTPUT_ATTENTIONS: output_attentions,  # Output attention weights
                RETURN_ATTENTION_OUTPUT: return_attention_output,  # Self-attention layer output
                RETURN_FFNN_OUTPUT: return_feed_forward_output
            }
            #

            return kwargs


class LayersWrapper(ModuleWrapper):
    _module_name: str = 'layer modules'
    module_output: str = 'layers_output'

    _layer_dtype: Type[ModuleWrapper] = LayerWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wrappers
        if not isinstance(self._module, nn.ModuleList):
            raise TypeError('Layers must be encapsulated in a `ModuleList` object')
        self._layer_wrappers: nn.ModuleList = nn.ModuleList(
            self._layer_dtype(layer, super_wrapper=self) for layer in self._module
        )

    @property
    def is_wrapping(self) -> bool:
        return any(layer_wrapper.is_wrapping for layer_wrapper in self._layer_wrappers)

    def enable_wrapper(self):
        if not self.is_wrapping:
            for layer_wrapper in self._layer_wrappers:
                layer_wrapper.enable_wrapper()

    def disable_wrapper(self):
        if self.is_wrapping:
            for layer_wrapper in self._layer_wrappers:
                layer_wrapper.disable_wrapper()

    @property
    def layers(self):
        return self.base_module

    @property
    def layer_wrappers(self):
        return self._layer_wrappers

    @property
    def layers_iterator(self):
        return self.base_module

    def get_layer_wrappers_iterator(self) -> Iterable:
        return self.layer_wrappers

    def _use_dynamic_cache(self) -> bool:
        return all(isinstance(layer, SHARED_STRUCTURE_LAYERS) for layer in self.layers_iterator)

    def _init_state(
            self,
            embeddings: torch.FloatTensor,
            use_cache: bool = False,
            output_attentions: bool = False,  # Output attention weights
            output_hidden_states: bool = False,
            return_attention_output: bool = False,  # Self-attention layer output
            return_feed_forward_output: bool = False,
            **kwargs
    ):
        # Current hidden state
        kwargs[CURR_HIDDEN_STATE] = embeddings
        # Hidden states
        if output_hidden_states:
            kwargs[HIDDEN_STATES] = [embeddings]
        # Attention weights
        if output_attentions:
            kwargs[ATTN_WEIGHTS] = list()
        # Cache
        if use_cache:
            kwargs[CACHE] = None if self._use_dynamic_cache() else list()
        # Attention output
        if return_attention_output:
            kwargs[ATTN_OUTPUTS] = list()
        # FFNN output
        if return_feed_forward_output:
            kwargs[FFNN_OUTPUTS] = list()

        kwargs |= {
            USE_CACHE: use_cache,
            OUTPUT_ATTENTIONS: output_attentions,  # Output attention weights
            OUTPUT_HIDDEN_STATES: output_hidden_states,
            RETURN_ATTENTION_OUTPUT: return_attention_output,  # Self-attention layer output
            RETURN_FFNN_OUTPUT: return_feed_forward_output
        }

        return kwargs

    def _update_state(
            self,
            use_cache: bool = False,
            output_attentions: bool = False,  # Output attention weights
            output_hidden_states: bool = False,
            return_attention_output: bool = False,  # Self-attention layer output
            return_feed_forward_output: bool = False,
            layer_idx: int = -1,
            **kwargs
    ):
        layer_output = kwargs.pop(self._layer_dtype.module_output)
        # Current hidden state
        kwargs[CURR_HIDDEN_STATE] = layer_output[CURR_HIDDEN_STATE]
        # Hidden states
        if output_hidden_states:
            kwargs[HIDDEN_STATES].append(layer_output[CURR_HIDDEN_STATE])
        # Attention weights
        if output_attentions:
            kwargs[ATTN_WEIGHTS].append(layer_output[CURR_ATTN_WEIGHTS])
        # Cache
        if use_cache:
            if isinstance(kwargs[CACHE], DynamicCache) or isinstance(
                    self.super_wrapper.base_model, SHARED_STRUCTURE_MODELS
            ):
                kwargs[CACHE] = layer_output[CURR_KEY_VALUE]
            else:
                kwargs[CACHE].append(layer_output[CURR_KEY_VALUE])
        # Attention output
        if return_attention_output:
            kwargs[ATTN_OUTPUTS].append(layer_output[ATTN_OUTPUT])
        # FFNN output
        if return_feed_forward_output:
            kwargs[FFNN_OUTPUTS].append(layer_output[FFNN_OUTPUT])

        kwargs |= {
            USE_CACHE: use_cache,
            OUTPUT_ATTENTIONS: output_attentions,  # Output attention weights
            OUTPUT_HIDDEN_STATES: output_hidden_states,
            RETURN_ATTENTION_OUTPUT: return_attention_output,  # Self-attention layer output
            RETURN_FFNN_OUTPUT: return_feed_forward_output
        }

        return kwargs

    def _pre_process_input(
            self, *args, valid_mask: Optional[torch.LongTensor] = None, **kwargs
    ):
        #
        if isinstance(self.super_wrapper.base_model, GPT2PreTrainedModel):
            if valid_mask is not None:
                attention_mask = valid_mask.view(kwargs[BATCH_SIZE], -1)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=kwargs[DTYPE])
                attention_mask = (1.0 - attention_mask) * torch.finfo(kwargs[DTYPE]).min
            else:
                attention_mask = None
        elif isinstance(self.super_wrapper.base_model, (GemmaPreTrainedModel, LlamaPreTrainedModel)):
            attention_mask = self.super_wrapper.base_model._update_causal_mask(valid_mask, kwargs[EMBEDDINGS])
            # TODO find better solution
            if attention_mask.size()[-2] != kwargs[SEQ_LENGTH] or attention_mask.size()[-1] != kwargs[SEQ_LENGTH]:
                attention_mask = attention_mask[..., :kwargs[SEQ_LENGTH], :kwargs[SEQ_LENGTH]]
        elif isinstance(self.super_wrapper.base_model, MistralPreTrainedModel):
            if valid_mask is not None and self.super_wrapper.base_model._attn_implementation == 'flash_attention_2' and kwargs[BATCH_SIZE] > 1:
                if valid_mask[:, -1].sum().item() != kwargs[BATCH_SIZE]:
                    raise ValueError('`padding_side=\'right\'` is not with the Flash Attention version of Mistral')
            if self.super_wrapper.base_model._attn_implementation == 'flash_attention_2':
                # 2d mask is passed through the layers
                attention_mask = valid_mask if (valid_mask is not None and 0 in valid_mask) else None
            elif self.super_wrapper.base_model._attn_implementation == 'sdpa' and not kwargs[OUTPUT_ATTENTIONS]:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    valid_mask,
                    (kwargs[BATCH_SIZE], kwargs[SEQ_LENGTH]),
                    kwargs[EMBEDDINGS],
                    kwargs[PREFIX_LENGTH],
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    valid_mask,
                    (kwargs[BATCH_SIZE], kwargs[SEQ_LENGTH]),
                    kwargs[EMBEDDINGS],
                    kwargs[PREFIX_LENGTH],
                    sliding_window=self.super_wrapper.base_model.config.sliding_window,
                )
        else:
            raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
        #
        kwargs |= {ATTENTION_MASK: attention_mask}

        return kwargs

    def _wrapped_forward(self, **kwargs):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            output = layer_wrapper.forward(layer_idx=layer_idx, **output)
            # Update model state
            output = self._update_state(**output)

        return output

    def _post_process_output(
            self, base_model_output: bool = False, current_hidden_state: Optional[torch.FloatTensor] = None, **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()
        if base_model_output:
            #
            current_hidden_state = kwargs.pop(CURR_HIDDEN_STATE)
            hidden_states = kwargs.pop(HIDDEN_STATES)
            attention_weights = kwargs.pop(ATTN_WEIGHTS)
            cache = kwargs.pop(CACHE)
            #
            output = current_hidden_state,
            if hidden_states:
                output += hidden_states,
            if attention_weights:
                output += attention_weights,
            if cache:
                output += cache,

            return output
        else:
            kwargs |= {self.module_output: current_hidden_state}

            return kwargs


class PreTrainedModelWrapper(PreTrainedModel, BaseWrapper):
    TASK_SPECIFIC_CONFIGS_KEY: str = 'task_specific_params'
    WRAPPER_CONFIGS_KEY: str = 'wrapper'

    _model_name: str = 'model'
    model_output: str = 'model_output'

    _auto_model_dtype: Optional[Type[PreTrainedModel]] = None

    # TODO fix-me this is a temporary solution to use non-eager attention
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            benchmarking: bool = False
    ):
        super().__init__(model.config)  # TODO fixme (find better solution)
        #
        self._model: PreTrainedModel = model
        self._tokenizer: PreTrainedTokenizer = tokenizer
        #
        self._banchmarking: bool = benchmarking

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            model_args: Optional[Tuple] = None,
            model_kwargs: Optional[Dict] = None,
            lora_config: Optional[LoraConfig] = None,
            tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
            tokenizer_args: Optional[Tuple] = None,
            tokenizer_kwargs: Optional[Dict] = None,
            **wrapper_kwargs
    ):
        #
        model_args = model_args if model_args else tuple()
        model_kwargs = model_kwargs if model_kwargs else dict()
        tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path else pretrained_model_name_or_path
        tokenizer_args = tokenizer_args if tokenizer_args else tuple()
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else dict()
        #
        model_kwargs['attn_implementation'] = 'eager'  # Make better version
        task_specific_configs = model_kwargs.get(cls.TASK_SPECIFIC_CONFIGS_KEY, dict())
        model_kwargs[cls.TASK_SPECIFIC_CONFIGS_KEY] = task_specific_configs | {cls.WRAPPER_CONFIGS_KEY: wrapper_kwargs}
        #
        model = cls._auto_model_dtype.from_pretrained(
                pretrained_model_name_or_path, *model_args, **model_kwargs
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *tokenizer_args, **tokenizer_kwargs
        )
        if lora_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)  # TODO fix gradient checkpointing issue
            model = get_peft_model(model, lora_config)
        wrapper = cls(model, tokenizer)

        return wrapper

    def save_pretrained(self, *args, **kwargs):
        self.base_model.save_pretrained(*args, **kwargs)

    @property
    def base_model(self) -> PreTrainedModel:
        # logger.warning(
        #     f'The returned base {self._model_name} may be modified and may have internal wrappers still enabled.'
        # )
        if isinstance(self._model, PeftModel):
            return self._model.base_model.model
        else:
            return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def is_training(self) -> bool:
        return self.base_model.training

    @property  # TODO fixme (not updating)
    def is_benchmarking(self) -> bool:
        return self._banchmarking

    @property
    def gradient_checkpointing(self):
        return self.base_model.gradient_checkpointing

    def eval(self):
        self._model = self._model.eval()

        return self

    def train(self, mode: bool = True):
        self._model = self._model.train(mode=mode)

        return self

    def enable_benchmarking(self):
        self._benchmarking = True

    def disable_benchmarking(self):
        self._benchmarking = False

    def _wrapped_forward(self, **kwargs):
        # Model forward
        model_output = self.base_model.forward(**kwargs)
        # Extend input with module output
        output = kwargs | {self.model_name: model_output}

        return output

    def _post_process_output(self, base_model_output: bool = False, **kwargs):
        if self.base_model_ouput:
            return kwargs[self.model_output]
        else:
            return kwargs

    def forward(self, *args, base_model_output: bool = False, **kwargs):
        self.enable_wrapper()
        # Pre-process input
        kwargs = self._pre_process_input(*args, **kwargs)
        # Apply layer transformation
        output = self._wrapped_forward(**kwargs)
        # Post-process output
        output = self._post_process_output(base_model_output=base_model_output, **output)

        return output


class TransformerWrapper(PreTrainedModelWrapper):
    _model_name: str = 'transformer model'
    model_output: str = 'transformer_output'

    _auto_model_dtype: Optional[Type[PreTrainedModel]] = AutoModel

    _embedding_dtype: Type[ModuleWrapper] = EmbeddingWrapper
    _layers_dtype: Type[ModuleWrapper] = LayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attribute names
        self._embedding_attr: TransformerEmbeddingAttr = self._get_embedding_attr()
        self._position_embedding_attr: Optional[TransformerPositionEmbeddingAttr] = self._get_position_embedding_attr()
        self._layers_attr: TransformerLayersAttr = self._get_layers_attr()
        self._norm_attr: TransformerNormAttr = self._get_norm_attr()
        # Wrappers
        self._embedding_wrapper = self._embedding_dtype(
            getattr(self._model, self._embedding_attr.value),
            super_wrapper=self,
            position_embeddings=getattr(
                self._model, self._position_embedding_attr.value
            ) if self._position_embedding_attr is not None else None
        )
        self._layers_wrapper = self._layers_dtype(getattr(self._model, self._layers_attr.value), super_wrapper=self)

    @property
    def is_embedding_wrapping(self):
        return isinstance(getattr(self.base_model, self._embedding_attr.value), self._embedding_dtype)

    @property
    def are_layers_wrapping(self):
        return isinstance(getattr(self.base_model, self._layers_attr.value), self._layers_dtype)

    @property
    def is_wrapping(self):
        return self.is_embedding_wrapping or self.are_layers_wrapping  # TODO Decide for 'or' or 'and'

    def enable_embedding_wrapper(self):
        if not self.is_embedding_wrapping:
            setattr(self.base_model, self._embedding_attr.value, self.embedding_wrapper)

    def enable_layers_wrapper(self):
        if not self.are_layers_wrapping:
            self.layers_wrapper.enable_wrapper()
            setattr(self.base_model, self._layers_attr.value, self.layers_wrapper)

    def enable_wrapper(self):
        if not self.is_wrapping:
            self.enable_embedding_wrapper()
            self.enable_layers_wrapper()

    def disable_embedding_wrapper(self):
        if self.is_embedding_wrapping:
            setattr(self.base_model, self._embedding_attr.value, self.embedding_wrapper.base_module)

    def disable_layers_wrapper(self):
        if self.are_layers_wrapping:
            self.layers_wrapper.disable_wrapper()
            setattr(self.base_model, self._layers_attr.value, self.layers_wrapper.base_module)

    def disable_wrapper(self):
        if self.is_wrapping:
            self.disable_embedding_wrapper()
            self.disable_layers_wrapper()

    def _get_embedding_attr(self) -> TransformerEmbeddingAttr:
        return _get_module_attr_name(self.base_model, TransformerEmbeddingAttr)

    def _get_position_embedding_attr(self) -> Optional[TransformerPositionEmbeddingAttr]:
        try:
            return _get_module_attr_name(self.base_model, TransformerPositionEmbeddingAttr)
        except ValueError:
            return None

    def _get_layers_attr(self) -> TransformerLayersAttr:
        return _get_module_attr_name(self.base_model, TransformerLayersAttr)

    def _get_norm_attr(self) -> TransformerNormAttr:
        return _get_module_attr_name(self.base_model, TransformerNormAttr)

    @property
    def embedding(self):
        return self.embedding_wrapper.base_module

    @property
    def embedding_wrapper(self):
        return self._embedding_wrapper

    @property
    def layers(self):
        return self.layers_wrapper.base_module

    @property
    def layers_wrapper(self):
        return self._layers_wrapper

    @property
    def norm(self) -> nn.Module:
        return getattr(self.base_model, self._norm_attr.value)

    def _pre_process_input(
            self,
            *args,
            input_ids: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[DynamicCache, List[Tuple[torch.FloatTensor, torch.FloatTensor]]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        #
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.is_training and use_cache:
            logger.warning(
                '`use_cache=True` is incompatible with gradient checkpointing. '
                'Setting `use_cache = False` and `past_key_values = None`.'
            )
            use_cache = False
            past_key_values = None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #
        dtype = self.base_model.dtype
        # Input
        if (input_ids is not None) and (input_embeds is not None):
            raise ValueError('Models accept either `input_ids` or `inputs_embeds`, not both.')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.size()
            device: torch.device = input_ids.device
        elif input_embeds is not None:
            batch_size, seq_length, _ = input_embeds.size()
            device: torch.device = input_ids.device
        else:
            raise ValueError('One between `input_ids` or `inputs_embeds` must be specified.')
        # Cache
        if past_key_values is None:
            if isinstance(self.base_model, GPT2PreTrainedModel):
                past_key_values = [None] * self.config.num_hidden_layers
            elif isinstance(self.base_model, SHARED_STRUCTURE_MODELS):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            else:
                raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
            prefix_length = 0
        else:
            if isinstance(self.base_model, GPT2PreTrainedModel) and all(pkv is not None for pkv in past_key_values):
                prefix_length = past_key_values[0][0].size(-2)
            elif isinstance(self.base_model, SHARED_STRUCTURE_MODELS):
                if not isinstance(past_key_values, Cache):
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                prefix_length = past_key_values.get_usable_length(seq_length)
            else:
                raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
        # Positions
        if position_ids is not None:
            if isinstance(self.base_model, (GPT2PreTrainedModel, GemmaPreTrainedModel, LlamaPreTrainedModel)):
                position_ids = position_ids.unsqueeze(0)
            elif isinstance(self.base_model, MistralPreTrainedModel):
                position_ids = position_ids.view(-1, seq_length).long()
            else:
                raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
        else:
            if isinstance(self.base_model, (GPT2PreTrainedModel, GemmaPreTrainedModel, LlamaPreTrainedModel)):
                position_ids = torch.arange(
                    prefix_length, prefix_length + seq_length, dtype=torch.long, device=device
                ).unsqueeze(0)
            elif isinstance(self.base_model, MistralPreTrainedModel):
                position_ids = torch.arange(
                    prefix_length, prefix_length + seq_length, dtype=torch.long, device=device
                ).unsqueeze(0).view(-1, seq_length)
            else:
                raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
        #
        kwargs |= {
            INPUT_IDS: input_ids,
            EMBEDDINGS: input_embeds,
            VALID_MASK: attention_mask,
            POSITION_IDS: position_ids,
            PAST_KEY_VALUES: past_key_values,
            USE_CACHE: use_cache,
            OUTPUT_ATTENTIONS: output_attentions,
            OUTPUT_HIDDEN_STATES: output_hidden_states,
            RETURN_DICT: return_dict,
            BATCH_SIZE: batch_size,
            PREFIX_LENGTH: prefix_length,
            SEQ_LENGTH: seq_length,
            DTYPE: dtype,
            DEVICE: device
        }

        return kwargs

    def _wrapped_forward(self, **kwargs):
        # Embed input
        output = self.embedding_wrapper.forward(**kwargs)
        # Process embeddings with transformer layers
        output = self.layers_wrapper.forward(**output)
        # Apply last normalisation
        output_hidden_state = self.norm.forward(output.pop(self.layers_wrapper.module_output))
        # Extend output with normalised last hidden state
        output |= {self.model_output: output_hidden_state}

        return output

    def _post_process_output(
            self,
            base_model_output: bool = False,
            cache: Optional[Union[DynamicCache, List[Tuple[torch.FloatTensor, torch.FloatTensor]]]] = None,
            hidden_states: Optional[List[torch.FloatTensor]] = None,
            attention_weights: Optional[List[torch.FloatTensor]] = None,
            return_dict: bool = True,
            **kwargs
    ):
        # TODO expand supported models (e.g., GPT-2 uses output with past and cross-attention)
        #
        if cache is not None and isinstance(cache, DynamicCache):
            cache = cache.to_legacy_cache()
        if base_model_output:
            if hidden_states is not None:
                logger.warning(
                    'Note: the last tensor in the output `hidden_states` is the non-normalised tensor `last_hidden_state`.'
                )
            if return_dict:
                if isinstance(self.base_model, GPT2PreTrainedModel):
                    return BaseModelOutputWithPastAndCrossAttentions(
                        last_hidden_state=kwargs[self.model_output],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                elif isinstance(self.base_model, SHARED_STRUCTURE_MODELS):
                    return BaseModelOutputWithPast(
                        last_hidden_state=kwargs[self.model_output],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                else:
                    raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
            else:
                return tuple(
                    v for v in [kwargs[self.model_output], cache, hidden_states, attention_weights] if v is not None
                )
        else:
            output_hidden_state = kwargs.pop(self.model_output)
            kwargs |= {
                OUT_HIDDEN_STATE: output_hidden_state,
                CACHE: cache,
                HIDDEN_STATES: hidden_states,
                ATTN_WEIGHTS: attention_weights,
                RETURN_DICT: return_dict
            }

            return kwargs

    # TODO implement other PreTrainedModel methods


class LMHeadWrapper(ModuleWrapper):
    module_attr = LMHeadAttr

    def _wrapped_forward(self, output_hidden_state: Optional[torch.tensor] = None, **kwargs):
        if output_hidden_state is None:
            raise ValueError()
        #
        logits = self.base_module.forward(output_hidden_state)
        #
        output = kwargs | {self.module_output: logits, OUT_HIDDEN_STATE: output_hidden_state}

        return output


class CausalLMWrapper(PreTrainedModelWrapper):
    _model_name: str = 'causal language model'
    model_output: str = 'causal_language_model_output'

    _auto_model_dtype: Optional[Type[PreTrainedModel]] = AutoModelForCausalLM

    _transformer_dtype: Type[PreTrainedModelWrapper] = TransformerWrapper
    _lm_head_dtype: Type[ModuleWrapper] = LMHeadWrapper

    lm_loss: str = 'lm_loss'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attribute names
        self._transformer_attr: LMTransformerAttr = self._get_transformer_attr()
        self._lm_head_attr: LMHeadAttr = self._get_lm_head_attr()
        # Wrappers
        self._transformer_wrapper: TransformerWrapper = self._transformer_dtype(
            getattr(self._model, self._transformer_attr.value), self._tokenizer
        )
        self._lm_head_wrapper: LMHeadWrapper = self._lm_head_dtype(getattr(self._model, self._lm_head_attr.value))

    def _get_transformer_attr(self) -> LMTransformerAttr:
        return _get_module_attr_name(self._model, LMTransformerAttr)

    def _get_lm_head_attr(self) -> LMHeadAttr:
        return _get_module_attr_name(self._model, LMHeadAttr)

    @property
    def is_transformer_wrapping(self):
        return isinstance(getattr(self.base_model, self._transformer_attr.value), self._transformer_dtype)

    @property
    def is_lm_head_wrapping(self):
        return isinstance(getattr(self.base_model, self._lm_head_attr.value), self._lm_head_dtype)

    @property
    def is_wrapping(self):
        return self.is_transformer_wrapping or self.is_lm_head_wrapping

    def enable_transformer_wrapper(self):
        if not self.is_transformer_wrapping:
            self.transformer_wrapper.enable_wrapper()
            setattr(self.base_model, self._transformer_attr.value, self.transformer_wrapper)

    def enable_lm_head_wrapper(self):
        if not self.is_lm_head_wrapping:
            setattr(self.base_model, self._lm_head_attr.value, self.lm_head_wrapper)

    def enable_wrapper(self):
        if not self.is_wrapping:
            self.enable_transformer_wrapper()
            self.enable_lm_head_wrapper()

    @property
    def transformer(self) -> PreTrainedModel:
        return self._transformer_wrapper.base_model

    @property
    def transformer_wrapper(self):
        return self._transformer_wrapper

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head_wrapper.base_module

    @property
    def lm_head_wrapper(self):
        return self._lm_head_wrapper
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable()

    def _wrapped_forward(self, **kwargs):
        #
        output = self.transformer_wrapper.forward(**kwargs)
        # Apply last normalisation
        logits = self.lm_head_wrapper.forward(**output).pop(self._lm_head_wrapper.module_output)
        # Extend output with normalised last hidden state
        output |= {self.model_output: logits}

        return output

    def _pre_process_input(self, *args, **kwargs):
        if len(args):
            input_ids, *_ = args
            return super()._pre_process_input(*args, input_ids=input_ids, **kwargs)
        else:
            return super()._pre_process_input(*args, **kwargs)

    def _post_process_output(
            self,
            base_model_output: bool = False,
            labels: Optional[torch.LongTensor] = None,
            cache: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
            hidden_states: Optional[List[torch.FloatTensor]] = None,
            attention_weights: Optional[List[torch.FloatTensor]] = None,
            return_dict: bool = True,
            **kwargs
    ):
        base_model_output = base_model_output or self._benchmarking
        #
        if base_model_output:
            if hidden_states is not None:
                logger.warning(
                    'Note: the last tensor in the output `hidden_states` is the non-normalised tensor `last_hidden_state`.'
                )
            if return_dict:
                if isinstance(self.base_model, GPT2PreTrainedModel):
                    return CausalLMOutputWithCrossAttentions(
                        loss=kwargs.get(self.lm_loss),
                        logits=kwargs[self.model_output],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                elif isinstance(self.base_model, SHARED_STRUCTURE_MODELS):
                    return CausalLMOutputWithPast(
                        loss=kwargs.get(self.lm_loss),
                        logits=kwargs[self.model_output],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                else:
                    raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')
            else:
                return tuple(
                    v for v in [
                        kwargs.get(self.lm_loss), kwargs[self.model_output], cache, hidden_states, attention_weights
                    ] if v is not None
                )
        else:
            #
            logits = kwargs.pop(self.model_output)
            if labels:
                shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
                loss = F.cross_entropy(shift_logits, shift_labels)
            else:
                loss = None
            #
            kwargs |= {
                LOGITS: logits,
                LOSS: loss,
                CACHE: cache,
                HIDDEN_STATES: hidden_states,
                ATTN_WEIGHTS: attention_weights,
                RETURN_DICT: return_dict
            }

            return kwargs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        return_inner_states = return_inner_states or self._benchmarking
        #
        generate_output = super().generate(*args, **kwargs)
        # Re-run through layers to collect all data  # TODO find better solution
        if return_inner_states:
            #
            return self.forward(
                input_ids=generate_output,
                **{
                    k: kwargs.get(k) for k in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())
                    if k not in {'args', 'kwargs', 'self', 'base_model_output'}
                },
                return_dict=True,
                output_attentions=True,
                use_cache=True,
                output_hidden_states=True,
                return_attention_output=True,  # Self-attention layer output
                return_feed_forward_output=True
            ) | {OUTPUT_IDS: generate_output}
        else:
            return generate_output

    def prepare_inputs_for_generation(self, *args, base_model_output: bool = True, **kwargs):
        # TODO fix Llama and Gemma generation issue
        self.enable_wrapper()
        #
        inputs = self.base_model.prepare_inputs_for_generation(
            *args, **kwargs
        ) | {'base_model_output': base_model_output}

        return inputs

    # TODO implement other PreTrainedModel methods
