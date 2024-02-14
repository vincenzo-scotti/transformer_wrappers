from typing import Type

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


class RecursiveModelWrapper(PreTrainedModelWrapper):
    # TODO
    ...


class RecursiveModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = RecursiveModelWrapper

