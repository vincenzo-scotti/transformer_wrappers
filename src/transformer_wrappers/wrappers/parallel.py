from itertools import batched

from typing import Type, Optional, Iterable, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


__all__ = ['ParallelModelWrapper', 'ParallelModelWrapperForCausalLMWrapper']


class ParallelModelWrapper(PreTrainedModelWrapper):
    RATE: str = 'rate'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate: int = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(self.RATE, 1)

    def _layers_iterator(self, *args, rate: int = 1, **kwargs) -> Iterable:
        return batched(self.layers, rate)

    def _layer_wrapped_forward(
            self,
            idx: int, layers: Iterable[nn.Module],
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, cache, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            rate: int = 1, **kwargs
    ) -> Tuple[
        torch.FloatTensor,
        Optional[Iterable[torch.FloatTensor]],
        Optional[Iterable[torch.FloatTensor]],
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]]
    ]:
        #
        cumulative_hidden_states = hidden_states
        for idx, layer in enumerate(layers, start=idx * rate):
            tmp_hidden_states, hidden_states_stack, self_attentions_stack, cache = super()._layer_wrapped_forward(
                idx, layer,
                hidden_states, attention_mask, self_attentions_stack, hidden_states_stack,
                batch_size, seq_length, prefix_length, device,
                input_ids, input_embeddings, position_ids, cache, valid_mask,
                use_cache, output_attentions, output_hidden_states, return_dict,
                **kwargs
            )
            cumulative_hidden_states += tmp_hidden_states - hidden_states
        #
        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def forward(self, *args, rate: Optional[int] = None, **kwargs):
        # Check input validity
        rate = self.rate if rate is None else rate
        if (self.config.num_hidden_layers // rate) * rate != self.config.num_hidden_layers:
            raise ValueError('`rate` must be an integer divisor of `num_hidden_layers`')
        #
        super().forward(*args, **kwargs, rate=rate)


class ParallelModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = ParallelModelWrapper
