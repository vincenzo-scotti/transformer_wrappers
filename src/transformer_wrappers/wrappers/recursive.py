from itertools import cycle

from typing import Type, Iterable, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import DynamicCache

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


__all__ = ['RecursiveModelWrapper', 'RecursiveModelWrapperForCausalLMWrapper']


class RecursiveModelWrapper(PreTrainedModelWrapper):
    def _layers_iterator(self, *args, iterations: Optional[List[int]] = None, **kwargs) -> Iterable:
        for i, (n_iter, layer) in zip(iterations, cycle(self.layers)):
            if n_iter > 0:
                yield i % self.config.num_hidden_layers, layer

    def _layer_wrapped_forward(
            self,
            idx: int, args: Tuple[int, nn.Module],
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
        layer_idx, layer = args
        #
        hidden_states, hidden_states_stack, self_attentions_stack, cache = super()._layer_wrapped_forward(
            layer_idx, layer,
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, cache, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            **kwargs
        )
        #
        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def forward(self, *args, iterations: Optional[List[int]] = None, **kwargs):
        # TODO add alternative methods to pass number of iterations
        # Check input validity
        iterations = self.iterations if iterations is None else iterations
        if len(iterations) % self.config.num_hidden_layers != 0:
            raise ValueError('The elements if `iterations` must be a multiple of `num_hidden_layers`')
        #
        return super().forward(*args, **kwargs, iterations=iterations)


class RecursiveModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = RecursiveModelWrapper

    def prepare_inputs_for_generation(self, *args, iterations: Optional[List[int]] = None, **kwargs):
        return super().prepare_inputs_for_generation(*args, **kwargs) | {'iterations': iterations}
