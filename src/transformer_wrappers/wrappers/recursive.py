from typing import Type

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


class RecursiveModelWrapper(PreTrainedModelWrapper):
    # TODO
    # Stack iterations
    # Block iterations
    # block idxs
    # Iterations map

    ...


class RecursiveModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = RecursiveModelWrapper

