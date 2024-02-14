from typing import Type

from .base import PreTrainedModelWrapper, PreTrainedModelWrapperForCausalLM


class ParallelModelWrapper(PreTrainedModelWrapper):
    # TODO
    ...


class ParallelModelWrapperForCausalLMWrapper(PreTrainedModelWrapperForCausalLM):
    _wrapper_class: Type[PreTrainedModelWrapper] = ParallelModelWrapper
