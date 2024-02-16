from itertools import cycle

from typing import Type, Iterable, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers import logging

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


__all__ = ['RecursiveModelWrapper', 'RecursiveModelWrapperForCausalLMWrapper']


logger = logging.get_logger(__name__)


class RecursiveModelWrapper(PreTrainedModelWrapper):
    ITERATIONS: str = 'iterations'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations: Optional[List[int]] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(
            self.ITERATIONS, [1] * self.config.num_hidden_layers
        )

    def _layers_iterator(self, *args, iterations: Optional[List[int]] = None, **kwargs) -> Iterable:
        for i, (n_iter, layer) in enumerate(zip(iterations, cycle(self.layers))):
            if n_iter > 0:
                yield i % self.config.num_hidden_layers, layer

    def _layer_wrapped_forward(
            self,
            idx: int, args: Tuple[int, nn.Module],
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            iterations: Optional[List] = None, **kwargs
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
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            **kwargs
        )
        #
        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def preprocess_wrapper_params(self, iterations: Optional[Union[int, List[int]]] = None):
        # TODO add alternative methods to pass number of iterations
        # Check input validity
        iterations = self.iterations if iterations is None else iterations
        if isinstance(iterations, int):
            iterations = [iterations] * self.config.num_hidden_layers
        if len(iterations) % self.config.num_hidden_layers != 0:
            raise ValueError('The elements if `iterations` must be a multiple of `num_hidden_layers`')
        #
        return {self.ITERATIONS: iterations}

    def forward(self, *args, iterations: Optional[Union[int, List[int]]] = None, **kwargs):
        #
        kwargs |= self.preprocess_wrapper_params(iterations=iterations)
        return super().forward(*args, **kwargs)


class RecursiveModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = RecursiveModelWrapper

    def forward(self, *args, iterations: Optional[Union[int, List[int]]] = None, **kwargs):
        logger.warning('Do not use recursive model concurrently, it may lead to unforeseen behaviour.')
        old_attributes = self._update_wrapper_attributes(
            **self.wrapper.preprocess_wrapper_params(iterations=iterations)
        )
        output = super().forward(*args, **kwargs)
        _ = self._update_wrapper_attributes(**old_attributes)

        return output

    def prepare_inputs_for_generation(self, *args, iterations: Optional[List[int]] = None, **kwargs):
        return super().prepare_inputs_for_generation(*args, **kwargs) | {'iterations': iterations}
