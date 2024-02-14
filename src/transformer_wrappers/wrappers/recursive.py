from typing import Type, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DynamicCache

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


__all__ = ['RecursiveModelWrapper', 'RecursiveModelWrapperForCausalLMWrapper']


class RecursiveModelWrapper(PreTrainedModelWrapper):
    # TODO
    # Stack iterations
    # Block iterations
    # block idxs
    # Iterations map

    def _layers_iterator(self, *args, rate: int = 1, **kwargs) -> Iterable:
        for ...:
            yield layer_idx, layer

    def _layer_wrapped_forward(
            self,
            idx: int, ,
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


class RecursiveModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = RecursiveModelWrapper
