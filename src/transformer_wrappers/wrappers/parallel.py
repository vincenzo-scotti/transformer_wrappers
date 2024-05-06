import math

from itertools import batched, chain, repeat
from typing import Type, Optional, Iterable, Dict

import torch
from transformers import DynamicCache
from transformers import logging


from .base import SHARED_STRUCTURE_MODELS
from .base import ModuleWrapper, PreTrainedModelWrapper
from .base import LayerWrapper, LayersWrapper
from .base import TransformerWrapper, CausalLMWrapper

from .constants import *


__all__ = [
    'ParallelTransformerWrapper',
    'ParallelCausalLMWrapper'
]


P_BLOCKS: str = 'p_blocks'
P_RATE: str = 'p_rate'
BLOCK_PARALLEL: str = 'block_parallel'
ITERATIVE: str = 'iterative'
AVERAGE: str = 'average'
COMPENSATE_AVG: str = 'compensate_avg'

SKIP_ATTENTION: str = 'skip_attention'
SKIP_FFNN: str = 'skip_ffnn'


logger = logging.get_logger(__name__)


class ParallelLayerWrapper(LayerWrapper):
    module_output: str = 'parallel_layer_output'
    
    # TODO fixme
    def _attn_forward(
            self,
            current_hidden_state: Optional[torch.FloatTensor],
            add_attn_residual: bool = True,
            compensate_avg: bool = False,
            block_size: int = 1,
            **kwargs
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
        if compensate_avg:
            attention_output[self.attention_wrapper.module_output] *= block_size
        if add_attn_residual:
            current_hidden_state = attention_output[self.attention_wrapper.module_output] + residual
        else:
            current_hidden_state = attention_output[self.attention_wrapper.module_output]
        #
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            INTERMEDIATE_HIDDEN_STATE: current_hidden_state,
            ADD_ATTN_RESIDUAL: add_attn_residual,
            COMPENSATE_AVG: compensate_avg,
            self.attention_wrapper.module_output: attention_output
        }

        return output

    # TODO fixme
    def _ffnn_forward(
            self,
            current_hidden_state,
            add_ffnn_residual: bool = True,
            compensate_avg: bool = False,
            block_size: int = 1,
            **kwargs
    ):
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
        if compensate_avg:
            ffnn_output *= block_size
        if add_ffnn_residual:
            current_hidden_state = ffnn_output + residual  # TODO verify this
        else:
            current_hidden_state = ffnn_output
        # Extend input with module output
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_FFNN_RESIDUAL: add_ffnn_residual,
            COMPENSATE_AVG: compensate_avg,
            self.feed_forward_wrapper.module_output: ffnn_output
        }

        return output

    def _wrapped_forward(
            self,
            current_hidden_state: Optional[torch.tensor] = None,
            skip_attention: bool = False,
            skip_ffnn: bool = False,
            block_size: int = 1,
            **kwargs
    ):
        if skip_attention and skip_ffnn:
            raise ValueError()
        #
        output = kwargs | {CURR_HIDDEN_STATE: current_hidden_state}
        if skip_attention:
            output[self.attention_wrapper.module_output] = {
                self.attention_wrapper.module_output: None,
                self.attention_wrapper.attention_weights: None,
                self.attention_wrapper.key_value: None
            }
        else:
            output = self._attn_forward(block_size=block_size, **output)
        if skip_ffnn:
            output[self.feed_forward_wrapper.module_output] = None
        else:
            output = self._ffnn_forward(block_size=block_size, **output)
        #
        output |= {
            INPUT_HIDDEN_STATE: current_hidden_state,
            self.module_output: output.pop(CURR_HIDDEN_STATE),
            SKIP_ATTENTION: skip_attention,
            SKIP_FFNN: skip_ffnn
        }

        return output
        

class ParallelLayersWrapper(LayersWrapper):
    TMP_HIDDEN_STATE: str = 'tmp_hidden_state'
    
    _layer_dtype: Type[ModuleWrapper] = ParallelLayerWrapper

    def get_layer_wrappers_iterator(self, p_rate: int = 1, iterative: bool = False) -> Iterable:
        if iterative:
            return chain.from_iterable(
                repeat(block, len(block)) for block in batched(super().get_layer_wrappers_iterator(), p_rate)
            )
        else:
            return batched(super().get_layer_wrappers_iterator(), p_rate)

    def _update_state(
            self,
            use_cache: bool = False,
            output_attentions: bool = False,  # Output attention weights
            output_hidden_states: bool = False,
            return_attention_output: bool = False,  # Self-attention layer output
            return_feed_forward_output: bool = False,
            layer_idx: Optional[int] = None,
            add_attn_residual: bool = True,
            add_ffnn_residual: bool = True,
            skip_attention: bool = False,
            skip_ffnn: bool = False,
            block_parallel: bool = False,
            average: bool = True,
            **kwargs
    ):
        #
        layer_output: Optional[Dict] = kwargs.pop(self._layer_dtype.module_output, None)
        #
        if layer_output is not None:
            # Current layer output
            if average:
                kwargs[self.TMP_HIDDEN_STATE] = kwargs.get(self.TMP_HIDDEN_STATE, list())
            else:
                kwargs[self.TMP_HIDDEN_STATE] = kwargs.get(
                    self.TMP_HIDDEN_STATE, [kwargs[INPUT_HIDDEN_STATE].clone()]
                )
                if block_parallel:
                    kwargs[self.TMP_HIDDEN_STATE].append(layer_output[ATTN_OUTPUT].clone())
            kwargs[self.TMP_HIDDEN_STATE].append(layer_output.pop(CURR_HIDDEN_STATE))
            # Set Current hidden state as input of previous block
            kwargs[CURR_HIDDEN_STATE] = kwargs.pop(INPUT_HIDDEN_STATE)
            # Attention weights
            if output_attentions and layer_output[CURR_ATTN_WEIGHTS] is not None:
                kwargs[ATTN_WEIGHTS].append(layer_output[CURR_ATTN_WEIGHTS])
            # Cache
            if use_cache and not skip_attention:
                if isinstance(kwargs[CACHE], DynamicCache) or isinstance(
                        self.super_wrapper.base_model, SHARED_STRUCTURE_MODELS
                ):
                    kwargs[CACHE] = layer_output[CURR_KEY_VALUE]
                else:
                    kwargs[CACHE].append(layer_output[CURR_KEY_VALUE])
            # Attention output
            if return_attention_output and not skip_attention:
                kwargs[ATTN_OUTPUTS].append(layer_output[ATTN_OUTPUT])
            # FFNN output
            if return_feed_forward_output and not skip_ffnn:
                kwargs[FFNN_OUTPUTS].append(layer_output[FFNN_OUTPUT])
        else:
            # End of block
            # Current hidden state
            if average:
                kwargs[CURR_HIDDEN_STATE] = torch.stack(kwargs.pop(self.TMP_HIDDEN_STATE)).mean(dim=0)
            else:
                kwargs[CURR_HIDDEN_STATE] = torch.stack(kwargs.pop(self.TMP_HIDDEN_STATE)).sum(dim=0)
            # Hidden states
            if output_hidden_states and not skip_ffnn:
                kwargs[HIDDEN_STATES].append(kwargs[CURR_HIDDEN_STATE])

        kwargs |= {
            USE_CACHE: use_cache,
            OUTPUT_ATTENTIONS: output_attentions,  # Output attention weights
            OUTPUT_HIDDEN_STATES: output_hidden_states,
            RETURN_ATTENTION_OUTPUT: return_attention_output,  # Self-attention layer output
            RETURN_FFNN_OUTPUT: return_feed_forward_output
        }
        
        return kwargs

    def _wrapped_forward_iteration(
            self,
            layer_wrappers_block: Optional[Iterable] = None,
            p_blocks: int = 0,
            p_rate: int = 1,
            average: bool = True,
            block_parallel: bool = True,
            block_idx: int = -1,
            iterative: bool = False,
            **kwargs
    ):
        if layer_wrappers_block is None:
            raise ValueError()
        if p_blocks == 0:
            raise ValueError()
        if block_idx < 0:
            raise ValueError()
        #
        output = kwargs
        additional_kwargs = [{SKIP_FFNN: True}, {SKIP_ATTENTION: True}] if not block_parallel else [dict()]
        for add_kwargs in additional_kwargs:
            for block_layer_idx, layer_wrapper in enumerate(layer_wrappers_block):
                if p_rate == len(layer_wrappers_block) or not iterative:
                    layer_idx = block_idx * p_rate + block_layer_idx
                else:
                    layer_idx = (
                            (((block_idx // p_rate) * p_rate) * p_rate) +
                            ((block_idx % p_rate) * len(layer_wrappers_block)) +
                            block_layer_idx
                    )
                #
                original_layer_idx = None
                if isinstance(self.super_wrapper.base_model, SHARED_STRUCTURE_MODELS):
                    original_layer_idx = layer_wrapper.attention_wrapper.base_module.layer_idx
                    layer_wrapper.attention_wrapper.base_module.layer_idx = layer_idx
                return_attention_output = output.pop(RETURN_ATTENTION_OUTPUT, False)
                # Apply layer transformation
                output = layer_wrapper.forward(
                    layer_idx=layer_idx,
                    add_attn_residual=average or block_parallel,
                    add_ffnn_residual=average,
                    return_attention_output=return_attention_output or (block_parallel and not average),
                    block_size=len(layer_wrappers_block),
                    **add_kwargs,
                    **output
                )
                #
                if isinstance(self.super_wrapper.base_model, SHARED_STRUCTURE_MODELS):
                    layer_wrapper.attention_wrapper.base_module.layer_idx = original_layer_idx
                output[RETURN_ATTENTION_OUTPUT] = return_attention_output
                # Update model state
                output = self._update_state(block_parallel=block_parallel, average=average, **output)
            # Update model state
            output = self._update_state(
                average=average, **add_kwargs, **output
            )

        return output

    def _wrapped_forward(
            self, 
            p_blocks: int = 1, 
            p_rate: int = 1, 
            block_parallel: bool = True, 
            iterative: bool = True, 
            average: bool = True, 
            **kwargs
    ):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for block_idx, layer_wrappers_block in enumerate(
                self.get_layer_wrappers_iterator(p_rate=p_rate, iterative=iterative)
        ):
            # Compute block output
            output = self._wrapped_forward_iteration(
                layer_wrappers_block=layer_wrappers_block,
                p_blocks=p_blocks,
                p_rate=p_rate,
                average=average,
                block_parallel=block_parallel,
                block_idx=block_idx,
                iterative=iterative,
                **output
            )
        #
        output |= {
            P_BLOCKS: p_blocks, P_RATE: p_rate, BLOCK_PARALLEL: block_parallel, ITERATIVE: iterative, AVERAGE: average
        }

        return output


class ParallelTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[ModuleWrapper] = ParallelLayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.p_blocks: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(P_BLOCKS)
        self.p_rate: int = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(P_RATE, 1)
        self.block_parallel: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(BLOCK_PARALLEL, False)
        self.iterative: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(ITERATIVE, True)
        self.average: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(AVERAGE, True)
        self.compensate_avg: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(COMPENSATE_AVG, False)

    def _pre_process_input(
            self,
            *args,
            p_blocks: Optional[int] = None,
            p_rate: Optional[int] = None,
            block_parallel: Optional[bool] = None,
            iterative: Optional[bool] = None,
            average: Optional[bool] = None,
            compensate_avg: Optional[bool] = None,
            **kwargs
    ):
        kwargs = super()._pre_process_input(*args, **kwargs)
        #
        if p_blocks is not None and p_rate is not None:
            raise ValueError('Parallel wrappers accept either `p_blocks` or `p_rate`, not both.')
        p_blocks = p_blocks if p_blocks is not None else self.p_blocks
        if p_blocks is not None:
            # if self.config.num_hidden_layers % p_blocks != 0:
            #     raise ValueError('`p_blocks` must be an integer divisor of `num_hidden_layers`')
            # else:
            #     p_rate = self.config.num_hidden_layers // p_blocks
            p_rate = int(math.ceil(self.config.num_hidden_layers / p_blocks))
        p_rate = p_rate if p_rate is not None else self.p_rate
        # p_blocks = p_blocks if p_blocks is not None else self.config.num_hidden_layers // p_rate
        p_blocks = p_blocks if p_blocks is not None else int(math.ceil(self.config.num_hidden_layers / p_rate))
        # if p_rate * p_blocks != self.config.num_hidden_layers:
        #     raise ValueError('`p_rate` must be an integer divisor of `num_hidden_layers`')
        block_parallel = block_parallel if block_parallel is not None else self.block_parallel
        iterative = iterative if iterative is not None else self.iterative
        average = average if average is not None else self.average
        compensate_avg = compensate_avg if compensate_avg is not None else self.compensate_avg
        #
        kwargs |= {
            P_BLOCKS: p_blocks,
            P_RATE: p_rate,
            BLOCK_PARALLEL: block_parallel,
            ITERATIVE: iterative,
            AVERAGE: average,
            COMPENSATE_AVG: compensate_avg
        }

        return kwargs


class ParallelCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = ParallelTransformerWrapper

    def prepare_inputs_for_generation(
            self,
            *args,
            p_blocks: Optional[int] = None,
            p_rate: Optional[int] = None,
            block_parallel: Optional[bool] = None,
            iterative: Optional[bool] = None,
            average: Optional[bool] = None,
            compensate_avg: Optional[bool] = None,
            **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs) | {
            P_BLOCKS: p_blocks,
            P_RATE: p_rate,
            BLOCK_PARALLEL: block_parallel,
            ITERATIVE: iterative,
            AVERAGE: average,
            COMPENSATE_AVG: compensate_avg
        }

        return inputs

