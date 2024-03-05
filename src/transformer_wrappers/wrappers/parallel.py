from typing import Type, Optional, Iterable, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers import logging


from .base import ModuleWrapper, PreTrainedModelWrapper
from .base import LayerWrapper, LayersWrapper
from .base import TransformerWrapper, CausalLMWrapper


__all__ = [
    'ParallelTransformerWrapper',
    'ParallelCausalLMWrapper'
]


logger = logging.get_logger(__name__)


class ParallelLayerWrapper(LayerWrapper):
    ...


class ParallelLayersWrapper(LayersWrapper):
    ...


class ParallelTransformerWrapper(TransformerWrapper):
    BLOCKS: str = 'blocks'
    RATE: str = 'rate'

    _layers_dtype: Type[ModuleWrapper] = LayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.blocks: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(self.BLOCKS)
        self.rate: int = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(self.RATE, 1)

    def _layers_iterator(self, *args, rate: int = 1, **kwargs) -> Iterable:
        for i in range(0, len(self.layers) // rate):
            yield (layer for layer in self.layers[i * rate: (i + 1) * rate])

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
        if self.config.num_hidden_layers % rate != 0:
            raise ValueError('`rate` must be an integer divisor of `num_hidden_layers`')
        #
        kwargs |= {self.BLOCKS: blocks, self.RATE: rate}

        return kwargs


class ParallelCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[PreTrainedModelWrapper] = ParallelTransformerWrapper

    def prepare_inputs_for_generation(self, *args, blocks: Optional[int] = None, rate: Optional[int] = None, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs) | {self.BLOCKS: blocks, self.RATE: rate}

        return inputs

