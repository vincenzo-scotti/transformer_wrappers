import os

from enum import Enum
from typing import Union, Optional, Type, Tuple, Dict

import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


__all__ = ['PreTrainedModelWrapper', 'PreTrainedModelWrapperForCausalLM']


TASK_SPECIFIC_CONFIGS_KEY: str = 'task_specific_params'


class TransformerNNAttr(Enum):
    TRANSFORMER = 'transformer'
    MODEL = 'model'


class PreTrainedModelWrapper(PreTrainedModel):
    WRAPPER_CONFIGS_KEY: str = 'wrapper'

    _auto_model_class: Type[PreTrainedModel] = AutoModel

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer
    ):
        super().__init__(model.config)  # TODO fixme (find better solution)
        self._model: PreTrainedModel = model
        self._tokenizer: PreTrainedTokenizer = tokenizer

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            model_args: Optional[Tuple] = None,
            model_kwargs: Optional[Dict] = None,
            tokenizer_args: Optional[Tuple] = None,
            tokenizer_kwargs: Optional[Dict] = None,
            **wrapper_kwargs
    ):
        #
        model_args = model_args if model_args else tuple()
        model_kwargs = model_kwargs if model_kwargs else dict()
        tokenizer_args = tokenizer_args if tokenizer_args else tuple()
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else dict()
        #
        model_kwargs[TASK_SPECIFIC_CONFIGS_KEY] = model_kwargs.get(TASK_SPECIFIC_CONFIGS_KEY, dict()) | wrapper_kwargs
        #
        wrapper = cls(
            cls._auto_model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **model_kwargs
            ),
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, *tokenizer_args, **tokenizer_kwargs
            )
        )

        return wrapper

    def save_pretrained(self, *args, **kwargs):
        self._model.save_pretrained(*args, **kwargs)

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    # TODO implement other PreTrainedModel methods
    

class PreTrainedModelWrapperForCausalLM(PreTrainedModelWrapper):
    _auto_model_class: Type[PreTrainedModel] = AutoModelForCausalLM
    _wrapper_class: Type[PreTrainedModelWrapper] = PreTrainedModelWrapper

    def __init__(self, *args, **kwags):
        #
        super().__init__(*args, **kwags)
        self._transformer_nn_attr: TransformerNNAttr = self._get_transformer_nn_attr()
        self._wrapper = self._wrapper_class(
            getattr(self._model, self._transformer_nn_attr.value), self._tokenizer
        )

    def _get_transformer_nn_attr(self) -> TransformerNNAttr:
        #
        for attr in TransformerNNAttr:
            if hasattr(self.model, attr.value):
                return attr
        #
        raise ValueError('Unsupported Causal LM model type')

    def enable_wrapper(self):
        if not self.is_wrapping:
            setattr(self._model, self._transformer_nn_attr.value, self._wrapper)

    def disable_wrapper(self):
        if self.is_wrapping:
            setattr(self._model, self._transformer_nn_attr.value, self._wrapper.model)

    def save_pretrained(self, *args, **kwargs):
        self.disable_wrapper()
        self._model.save_pretrained(*args, **kwargs)

    @property
    def model(self) -> PreTrainedModel:
        return self._wrapper.model

    @property
    def lm_head(self) -> nn.Linear:
        return self._model.lm_head

    @property
    def wrapper(self):
        return self._wrapper

    @property
    def is_wrapping(self):
        return isinstance(getattr(self.model, self._transformer_nn_attr.value), self._wrapper_class)

    def forward(self, *args, **kwargs):
        self.enable_wrapper()
        self._model(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        self.enable_wrapper()
        self._model.prepare_inputs_for_generation(*args, **kwargs)
