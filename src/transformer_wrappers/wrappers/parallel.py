from typing import Type, Optional, Iterable, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers import logging


from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


__all__ = ['ParallelModelWrapper', 'ParallelModelWrapperForCausalLMWrapper']


logger = logging.get_logger(__name__)


class ParallelModelWrapper(PreTrainedModelWrapper):
    BLOCKS: str = 'blocks'
    RATE: str = 'rate'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(self.BLOCKS)
        self.rate: int = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(self.RATE, 1)

    def _layers_iterator(self, *args, rate: int = 1, **kwargs) -> Iterable:
        for i in range(0, len(self.layers) // rate):
            yield (layer for layer in self.layers[i * rate: (i + 1) * rate])

    def _layer_wrapped_forward(
            self,
            idx: int, layers: Iterable[nn.Module],
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            blocks: Optional[int] = None, rate: int = 1, **kwargs
    ) -> Tuple[
        torch.FloatTensor,
        Optional[Iterable[torch.FloatTensor]],
        Optional[Iterable[torch.FloatTensor]],
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]]
    ]:
        #
        cumulative_hidden_states = torch.empty_like(hidden_states)
        for idx, layer in enumerate(layers, start=idx * rate):
            tmp_hidden_states, hidden_states_stack, self_attentions_stack, cache = super()._layer_wrapped_forward(
                idx, layer,
                hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
                batch_size, seq_length, prefix_length, device,
                input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
                use_cache, output_attentions, output_hidden_states, return_dict,
                **kwargs
            )
            cumulative_hidden_states += tmp_hidden_states - hidden_states
            # cumulative_hidden_states = tmp_hidden_states
        cumulative_hidden_states += hidden_states
        #
        return cumulative_hidden_states, hidden_states_stack, self_attentions_stack, cache

    def preprocess_wrapper_params(self, blocks: Optional[int] = None, rate: Optional[int] = None):
        # Check input validity
        if blocks is not None and rate is not None:
            raise ValueError('Parallel wrappers accept either `blocks` or `rate`, not both.')
        blocks = self.blocks if blocks is None else blocks
        if blocks is not None:
            if self.config.num_hidden_layers % blocks != 0:
                raise ValueError('`blocks` must be an integer divisor of `num_hidden_layers`')
            else:
                rate = self.config.num_hidden_layers // blocks
        rate = self.rate if rate is None else rate
        if self.config.num_hidden_layers % rate != 0:
            raise ValueError('`rate` must be an integer divisor of `num_hidden_layers`')

        return {self.BLOCKS: blocks, self.RATE: rate}

    def forward(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        #
        kwargs |= self.preprocess_wrapper_params(blocks=blocks, rate=rate)
        return super().forward(*args, **kwargs)


class ParallelModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = ParallelModelWrapper

    def forward(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        logger.warning('Do not use parallel model concurrently, it may lead to unforeseen behaviour.')
        old_attributes = self._update_wrapper_attributes(
            **self.wrapper.preprocess_wrapper_params(blocks=blocks, rate=rate)
        )
        output = super().forward(*args, **kwargs)
        _ = self._update_wrapper_attributes(**old_attributes)

        return output

    def prepare_inputs_for_generation(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        return super().prepare_inputs_for_generation(*args, **kwargs) | {'blocks': blocks, 'rate': rate}

