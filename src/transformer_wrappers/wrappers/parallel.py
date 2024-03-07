from itertools import batched
from typing import Type, Optional, Iterable, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers import logging


from .base import ModuleWrapper, PreTrainedModelWrapper
from .base import LayerWrapper, LayersWrapper
from .base import TransformerWrapper, CausalLMWrapper

from .constants import *


__all__ = [
    'ParallelTransformerWrapper',
    'ParallelCausalLMWrapper'
]


BLOCKS: str = 'blocks'
RATE: str = 'rate'


logger = logging.get_logger(__name__)


class ParallelLayerWrapper(LayerWrapper):
    module_output: str = 'parallel_layer_output'
    
    def _wrapped_forward(self, **kwargs):
        return super()._wrapped_forward(add_ffnn_residual=False, **kwargs)
        

class ParallelLayersWrapper(LayersWrapper):
    TMP_LAYER_OUTPUTS: str = 'tmp_layer_outputs'
    
    _layer_dtype: Type[ModuleWrapper] = ParallelLayerWrapper

    def get_layer_wrappers_iterator(self, rate: int = 1) -> Iterable:
        return batched(super().get_layer_wrappers_iterator(), rate)

    def _update_state(
            self,
            use_cache: bool = False,
            output_attentions: bool = False,  # Output attention weights
            output_hidden_states: bool = False,
            return_attention_output: bool = False,  # Self-attention layer output
            return_feed_forward_output: bool = False,
            layer_idx: int = -1,
            end_of_block: bool = False,
            **kwargs
    ):
        layer_output = kwargs.pop(self._layer_dtype.module_output)
        #
        if end_of_block:
            # Current hidden state
            kwargs[CURR_HIDDEN_STATE] = kwargs[CURR_HIDDEN_STATE] + sum(kwargs.pop(self.TMP_LAYER_OUTPUTS))
            # Hidden states
            if output_hidden_states:
                kwargs[HIDDEN_STATES].append(kwargs[CURR_HIDDEN_STATE])
        else:
            # Current layer output
            kwargs[self.TMP_LAYER_OUTPUTS] = kwargs.get(self.TMP_LAYER_OUTPUTS, list())
            kwargs[self.TMP_LAYER_OUTPUTS].append(layer_output[CURR_HIDDEN_STATE])
            # Attention weights
            if output_attentions:
                kwargs[ATTN_WEIGHTS].append(layer_output[CURR_ATTN_WEIGHTS])
            # Cache
            if use_cache:
                if isinstance(kwargs[CACHE], DynamicCache):
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

    def _wrapped_forward(self, rate: int = 1, **kwargs):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for block_idx, layer_wrapper_block in enumerate(self.layer_wrappers_iterator(rate=rate)):
            for block_layer_idx, layer_wrapper in enumerate(layer_wrapper_block):
                layer_idx = block_idx * rate + block_layer_idx
                # Apply layer transformation
                output = layer_wrapper.forward(layer_idx=layer_idx, **output)
                # Update model state
                output = self._update_state(**output)
            # Update model state
            output = self._update_state(end_of_block=True, **output)
        #
        output |= {RATE: rate}

        return output


class ParallelTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[ModuleWrapper] = ParallelLayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.blocks: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(BLOCKS)
        self.rate: int = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(RATE, 1)

    def _pre_process_input(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        kwargs = super()._pre_process_input(*args, **kwargs)
        #
        if blocks is not None and rate is not None:
            raise ValueError('Parallel wrappers accept either `blocks` or `rate`, not both.')
        blocks = self.blocks if blocks is None else blocks
        if blocks is not None:
            if self.config.num_hidden_layers % blocks != 0:
                raise ValueError('`blocks` must be an integer divisor of `num_hidden_layers`')
            else:
                rate = self.config.num_hidden_layers // blocks
        rate = self.rate if rate is None else rate
        blocks = blocks if blocks is not None else self.config.num_hidden_layers // rate
        if rate * blocks != self.config.num_hidden_layers:
            raise ValueError('`rate` must be an integer divisor of `num_hidden_layers`')
        #
        kwargs |= {BLOCKS: blocks, RATE: rate}

        return kwargs


class ParallelCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[PreTrainedModelWrapper] = ParallelTransformerWrapper

    def prepare_inputs_for_generation(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs) | {BLOCKS: blocks, RATE: rate}

        return inputs

