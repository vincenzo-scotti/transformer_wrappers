import os
import warnings

from enum import Enum
from typing import Union, Optional, Type, Tuple, Dict, List, Iterable

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Model, LlamaModel, MistralModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
)
from transformers import logging


__all__ = [
    'EmbeddingWrapper',
    'AttentionWrapper',
    'FeedForwardWrapper',
    'LayerWrapper',
    # 'LayersWrapper',  # TODO
    'TransformerWrapper',
    'CausalLMWrapper'
]


logger = logging.get_logger(__name__)


class TransformerEmbeddingAttr(Enum):
    EMBED_TOKENS = 'embed_tokens'
    WTE = 'wte'


class LayerInitialNormAttr(Enum):
    INPUT_LAYERNORM = 'input_layernorm'
    LN_1 = 'ln_1'


class LayerAttentionAttr(Enum):
    SELF_ATTN = 'self_attn'
    ATTN = 'attn'


class LayerIntermediateNormAttr(Enum):
    INPUT_LAYERNORM = 'input_layernorm'
    LN_2 = 'ln_2'


class LayerFeedForwardAttr(Enum):
    MLP = 'mlp'


class TransformerLayersAttr(Enum):
    LAYERS = 'layers'
    H = 'h'


class TransformerNormAttr(Enum):
    NORM = 'norm'
    LN_F = 'ln_f'


class LMTransformerAttr(Enum):
    MODEL = 'model'
    TRANSFORMER = 'transformer'
    
    
class LMHeadAttr(Enum):
    LM_HEAD = 'lm_head'


AttrEnumTypes: Type = Union[
    LayerInitialNormAttr, LayerAttentionAttr, LayerIntermediateNormAttr, LayerFeedForwardAttr,
    TransformerEmbeddingAttr, TransformerLayersAttr, TransformerNormAttr, 
    LMTransformerAttr, LMHeadAttr
]


def _get_module_attr_name(model: nn.Module, attr_names: Type[AttrEnumTypes]):
    #
    for attr in attr_names:
        if hasattr(model, attr.value):
            return attr
    #
    raise ValueError(f'Unsupported module type `{type(model)}` for attribute `{attr_names}`.')


class BaseWrapper:
    @property
    def is_wrapping(self) -> bool:
        return True

    def enable_wrapper(self):
        pass

    def disable_wrapper(self):
        pass


class ModuleWrapper(nn.Module, BaseWrapper):
    _module_name: str = 'module'
    module_attr: Optional[Type[AttrEnumTypes]] = None

    def __init__(self, module: nn.Module):
        self._module: nn.Module = module
        super().__init__()

    @property
    def base_module(self) -> nn.Module:
        logger.warning(f'The returned base {self._module_name} may be modified.')
        self.disable_wrapper()
        return self._module

    def forward(self, *args, **kwargs):
        self.enable_wrapper()
        return self.module.forward(*args, **kwargs)


class EmbeddingWrapper(ModuleWrapper):
    _module_name: str = 'embedding module'
    module_attr = TransformerEmbeddingAttr


class AttentionWrapper(ModuleWrapper):
    _module_name: str = 'attention module'
    module_attr = LayerAttentionAttr


class FeedForwardWrapper(ModuleWrapper):
    _module_name: str = 'feed forward module'
    module_attr = LayerFeedForwardAttr


class LayerWrapper(ModuleWrapper):
    _module_name: str = 'layer module'

    _attention_dtype: Type[ModuleWrapper] = AttentionWrapper
    _feed_forward_dtype: Type[ModuleWrapper] = FeedForwardWrapper

    def __init__(self, layer: nn.Module):
        super().__init__(layer)
        # Attribute names
        self._initial_norm_attr: LayerInitialNormAttr = self._get_initial_norm_attr()
        self._attention_attr: LayerAttentionAttr = self._get_attention_attr()
        self._intermediate_norm_attr: LayerIntermediateNormAttr = self._get_intermediate_norm_attr()
        self._feed_forward_attr: LayerFeedForwardAttr = self._get_feed_forward_attr()
        # Wrappers
        self._attention_wrapper = self._attention_dtype(getattr(self._module, self._attention_attr.value))
        self._feed_forward_wrapper = self._feed_forward_dtype(getattr(self._module, self._feed_forward_attr.value))

    @property
    def is_attention_wrapping(self):
        return isinstance(getattr(self.base_module, self._attention_attr.value), self._attention_dtype)

    @property
    def is_feed_forward_wrapping(self):
        return isinstance(getattr(self.base_module, self._feed_forward_attr.value), self._feed_forward_dtype)

    @property
    def is_wrapping(self):
        return self.is_attention_wrapping or self.is_feed_forward_wrapping  # TODO Decide for 'or' or 'and'

    def enable_attention_wrapper(self):
        setattr(self.base_module, self._attention_attr.value, self.attention_wrapper)

    def enable_feed_forward_wrapper(self):
        setattr(self.base_module, self._feed_forward_attr.value, self.feed_forward_wrapper)

    def enable_wrapper(self):
        if not self.is_wrapping:
            self.enable_attention_wrapper()
            self.enable_feed_forward_wrapper()

    def disable_attention_wrapper(self):
        setattr(self.base_module, self._attention_attr.value, self.attention_wrapper.base_module)

    def disable_feed_forward_wrapper(self):
        setattr(self.base_module, self._feed_forward_attr.value, self.feed_forward_wrapper.base_module)

    def disable_wrapper(self):
        if self.is_wrapping:
            self.disable_attention_wrapper()
            self.disable_feed_forward_wrapper()

    def _get_initial_norm_attr(self) -> LayerInitialNormAttr:
        return _get_module_attr_name(self.base_module, LayerInitialNormAttr)

    def _get_attention_attr(self) -> LayerAttentionAttr:
        return _get_module_attr_name(self.base_module, LayerAttentionAttr)

    def _get_intermediate_norm_attr(self) -> LayerIntermediateNormAttr:
        return _get_module_attr_name(self.base_module, LayerIntermediateNormAttr)

    def _get_feed_forward_attr(self) -> LayerFeedForwardAttr:
        return _get_module_attr_name(self.base_module, LayerFeedForwardAttr)

    @property
    def initial_norm(self) -> nn.Module:
        return getattr(self.base_module, self._layer_norm_attr)

    @property
    def attention(self):
        return self.attention_wrapper.base_module

    @property
    def attention_wrapper(self) -> AttentionWrapper:
        return self._attention_wrapper

    @property
    def intermediate_norm(self) -> nn.Module:
        return getattr(self.base_module, self._layer_norm_attr)

    @property
    def feed_forward(self):
        return self.feed_forward_wrapper.base_module

    @property
    def feed_forward_wrapper(self):
        return self._feed_forward_wrapper


class LayersWrapper(ModuleWrapper):
    _module_name: str = 'layer modules'

    _layer_dtype: Type[ModuleWrapper] = LayerWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wrappers
        if not isinstance(self._module, nn.ModuleList):
            raise TypeError('Layers must be encapsulated in a `ModuleList` object')
        self._layer_wrappers: nn.ModuleList = nn.ModuleList(self._layer_dtype(layer) for layer in self._module)

    @property
    def is_wrapping(self) -> bool:
        return any(layer_wrapper.is_wrapping for layer_wrapper in self._layer_wrappers)

    def enable_wrapper(self):
        for layer_wrapper in self._layer_wrappers:
            layer_wrapper.enable_wrapper()

    def disable_wrapper(self):
        for layer_wrapper in self._layer_wrappers:
            layer_wrapper.disable_wrapper()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def layers(self):
        return self.base_module

    @property
    def layer_wrappers(self):
        return self._layer_wrappers


class PreTrainedModelWrapper(PreTrainedModel, BaseWrapper):
    TASK_SPECIFIC_CONFIGS_KEY: str = 'task_specific_params'
    WRAPPER_CONFIGS_KEY: str = 'wrapper'

    _model_name: str = 'model'

    _auto_model_dtype: Optional[Type[PreTrainedModel]] = None

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer
    ):
        super().__init__(model.config)  # TODO fixme (find better solution)
        #
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
        task_specific_configs = model_kwargs.get(cls.TASK_SPECIFIC_CONFIGS_KEY, dict())
        model_kwargs[cls.TASK_SPECIFIC_CONFIGS_KEY] = task_specific_configs | {cls.WRAPPER_CONFIGS_KEY: wrapper_kwargs}
        #
        wrapper = cls(
            cls._auto_model_dtype.from_pretrained(
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
        self.base_model.save_pretrained(*args, **kwargs)

    @property
    def base_model(self) -> nn.Module:
        logger.warning(f'The returned base {self._model_name} may be modified.')
        self.disable_wrapper()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def forward(self, *args, **kwargs):
        self.enable_wrapper()
        return self.base_model.forward(*args, **kwargs)


class TransformerWrapper(PreTrainedModelWrapper):
    _model_name: str = 'transformer model'

    _auto_model_dtype: Optional[Type[PreTrainedModel]] = AutoModel

    _embedding_dtype: Type[ModuleWrapper] = EmbeddingWrapper
    _layers_dtype: Type[ModuleWrapper] = LayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attribute names
        self._embedding_attr: TransformerEmbeddingAttr = self._get_embedding_attr()
        self._layers_attr: TransformerLayersAttr = self._get_layers_attr()
        self._norm_attr: TransformerNormAttr = self._get_norm_attr()
        # Wrappers
        self._embedding_wrapper = self._embedding_dtype(getattr(self._model, self._embedding_attr.value))
        self._layers_wrapper = self._layers_dtype(getattr(self._module, self._layers_attr.value))

    @property
    def is_embedding_wrapping(self):
        return isinstance(getattr(self.base_model, self._embedding_attr.value), self._embedding_dtype)

    @property
    def are_layers_wrapping(self):
        return isinstance(getattr(self.base_model, self._layers_attr.value), self._layers_dtype)

    @property
    def is_wrapping(self):
        return self.is_embedding_wrapping or self.are_layers_wrapping  # TODO Decide for 'or' or 'and'

    def enable_embedding_wrapper(self):
        setattr(self.base_model, self._embedding_attr.value, self.embedding_wrapper)

    def enable_layers_wrapper(self):
        self.layers_wrapper.enable_wrapper()
        setattr(self.base_model, self._layers_attr.value, self.layers_wrapper)

    def enable_wrapper(self):
        if not self.is_wrapping:
            self.enable_attention_wrapper()
            self.enable_feed_forward_wrapper()

    def disable_embedding_wrapper(self):
        setattr(self.base_model, self._embedding_attr.value, self.embedding_wrapper.base_module)

    def disable_layers_wrapper(self):
        self.layers_wrapper.disable_wrapper()
        setattr(self.base_model, self._layers_attr.value, self.layers_wrapper.base_module)

    def disable_wrapper(self):
        if self.is_wrapping:
            self.disable_embedding_wrapper()
            self.disable_layers_wrapper()

    def _get_embedding_attr(self) -> TransformerEmbeddingAttr:
        return _get_module_attr_name(self.base_model, TransformerEmbeddingAttr)

    def _get_layers_attr(self) -> TransformerLayersAttr:
        return _get_module_attr_name(self.base_model, TransformerLayersAttr)

    def _get_norm_attr(self) -> TransformerNormAttr:
        return _get_module_attr_name(self.base_model, TransformerNormAttr)

    @property
    def embedding(self):
        return getattr(self.base_model, self._embedding_attr.value)

    @property
    def embedding_wrapper(self):
        return self._embedding_wrapper

    @property
    def layers(self):
        return getattr(self.base_model, self._layers_attr.value)

    @property
    def layers_wrapper(self):
        return self._layers_wrapper

    @property
    def norm(self) -> nn.Module:
        return getattr(self.base_model, self._transformer_norm_attr.value)

    @property
    def is_training(self) -> bool:
        return self._model.training

    @property
    def gradient_checkpointing(self):
        return self._model.gradient_checkpointing

    def preprocess_wrapper_params(self, **kwargs) -> Dict:
        raise NotImplementedError()

    def _model_specific_preprocessing(
            self,
            input_embeddings: torch.FloatTensor,
            position_ids: Optional[torch.LongTensor],
            valid_mask: torch.Tensor,
            past_key_values: Optional[
                Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]
            ],
            batch_size: int,
            seq_length: int,
            device: torch.device,
            dtype,
            use_cache: bool,
            output_attentions: bool
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.Tensor,
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]],
        Optional[Union[DynamicCache, Iterable[Tuple]]],
        int
    ]:
        if isinstance(self.base_model, GPT2Model):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
            # Cache (past key-values)
            past_key_values = past_key_values if past_key_values is not None else tuple(
                [None] * self.config.num_hidden_layers
            )
            cache = tuple() if use_cache else None
            # Position IDs
            prefix_length = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0
            position_ids = position_ids.unsqueeze(0) if position_ids is not None else torch.arange(
                prefix_length, prefix_length + seq_length, dtype=torch.long, device=device
            ).unsqueeze(0)
            # Attention mask
            if valid_mask is not None:
                attention_mask = valid_mask.view(batch_size, -1)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            else:
                attention_mask = None
            # Hidden states
            hidden_states = input_embeddings + self.base_model.wpe(position_ids)
            hidden_states = self.base_model.drop(hidden_states)
        elif isinstance(self.base_model, LlamaModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            # Cache (past key-values)
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            cache = None
            # Position IDs
            prefix_length = past_key_values.get_usable_length(seq_length)
            # Attention mask
            attention_mask = self._update_causal_mask(valid_mask, input_embeddings)
            # Hidden states
            hidden_states = input_embeddings
        elif isinstance(self.base_model, MistralModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
            # Cache (past key-values)
            if not isinstance(past_key_values, DynamicCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            cache = None
            # Position IDs
            prefix_length = past_key_values.get_usable_length(seq_length)
            position_ids = position_ids.view(-1, seq_length).long() if position_ids is not None else torch.arange(
                prefix_length, prefix_length + seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).view(-1, seq_length)
            # Attention mask
            if valid_mask is not None and self.base_model._attn_implementation == 'flash_attention_2' and use_cache:
                if valid_mask[:, -1].sum().item() != batch_size:
                    raise ValueError('`padding_side=\'right\'` is not with the Flash Attention version of Mistral')
            if self._attn_implementation == 'flash_attention_2':
                # 2d mask is passed through the layers
                attention_mask = valid_mask if (valid_mask is not None and 0 in valid_mask) else None
            elif self._attn_implementation == 'sdpa' and not output_attentions:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    valid_mask,
                    (batch_size, seq_length),
                    input_embeddings,
                    prefix_length,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    valid_mask,
                    (batch_size, seq_length),
                    input_embeddings,
                    prefix_length,
                    sliding_window=self.config.sliding_window,
                )
            # Hidden states
            hidden_states = input_embeddings
        else:
            raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')

        return hidden_states, attention_mask, position_ids, past_key_values, cache, prefix_length

    # NOTE I assumed there are no masked positions inside the single sequences
    def _pre_process_input(
            self,
            input_ids: Optional[torch.LongTensor],
            input_embeddings: Optional[torch.FloatTensor],
            position_ids: Optional[torch.LongTensor],
            past_key_values: Optional[
                Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]
            ],
            valid_mask: Optional[torch.Tensor],
            use_cache: Optional[bool],
            output_attentions: Optional[bool],
            output_hidden_states: Optional[bool],
            return_dict: Optional[bool],
    ):
        # TODO expand supported models (e.g., GPT-2 and BERT use token types and can have other inputs)
        #
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.is_training and use_cache:
            logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.')
            use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #
        if (input_ids is None) ^ (input_embeddings is not None):
            raise ValueError('Models accept either `input_ids` or `inputs_embeds`, not both.')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.size()
            device: torch.device = input_ids.device
        elif input_embeddings is not None:
            batch_size, seq_length, _ = input_embeddings.size()
            device: torch.device = input_ids.device
        else:
            raise ValueError('One between `input_ids` or `inputs_embeds` must be specified.')
        dtype = self.base_model.dtype
        input_embeddings = input_embeddings if input_embeddings is not None else self.embedding(input_ids)
        # Model-specific preprocessing
        (
            hidden_states, attention_mask, position_ids, past_key_values, cache, prefix_length
        ) = self._model_specific_preprocessing(
            input_embeddings, position_ids, valid_mask, past_key_values,
            batch_size, seq_length, device, dtype, use_cache, output_attentions
        )
        #
        self_attentions_stack: Optional[Tuple] = tuple() if output_attentions else None
        hidden_states_stack: Optional[Tuple[torch.FloatTensor]] = (hidden_states,) if output_hidden_states else None

        return (
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict
        )

    def _model_specific_layer_inputs(
            self,
            idx, layer,
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            **kwargs
    ) -> Tuple[Tuple, Dict]:
        #
        args = (hidden_states, )
        kwargs = dict() if self.gradient_checkpointing and self.is_training else {
            'attention_mask': attention_mask,
            'output_attentions': output_attentions,
            'use_cache': use_cache
        }
        #
        if isinstance(self.base_model, GPT2Model):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
            if self.gradient_checkpointing and self.is_training:
                args += (
                    None,
                    attention_mask,
                    None,
                    None,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:
                kwargs |= {'layer_past': past_key_values[idx]}  # TODO check this in presence of multiple iterations
        elif isinstance(self.base_model, LlamaModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            if self.gradient_checkpointing and self.is_training:
                args += (
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                kwargs |= {'past_key_value': past_key_values}
        elif isinstance(self.base_model, MistralModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
            if self.gradient_checkpointing and self.is_training:
                args += (
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache
                )
            else:
                kwargs |= {'past_key_value': past_key_values}
        else:
            raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')

        return args, kwargs

    def _update_state(
            self,
            layer_output,
            hidden_states_stack,
            self_attentions_stack,
            cache,
            use_cache: bool,
            output_attentions: bool,
            output_hidden_states: bool

    ) -> Tuple[
        torch.FloatTensor,
        Optional[Iterable[torch.FloatTensor]],
        Optional[Iterable[torch.FloatTensor]],
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]]
    ]:
        cache_update = self_attentions = None
        if use_cache and output_attentions:
            hidden_states, cache_update, self_attentions, *_ = layer_output
        elif use_cache:
            hidden_states, cache_update, *_ = layer_output
        elif output_attentions:
            hidden_states, self_attentions, *_ = layer_output
        else:
            hidden_states, *_ = layer_output
        #
        if use_cache is True:
            if isinstance(cache, DynamicCache):
                cache = cache_update
            else:
                cache += (cache_update,)
        #
        if output_hidden_states:
            hidden_states_stack += (hidden_states,)
        #
        if output_attentions:
            self_attentions_stack += (self_attentions,)

        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def _layer_wrapped_forward(
            self,
            idx: int, layer: nn.Module,
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            **kwargs
    ) -> Tuple[
        torch.FloatTensor,
        Optional[Iterable[torch.FloatTensor]],
        Optional[Iterable[torch.FloatTensor]],
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]]
    ]:
        input_args, input_kwargs = self._model_specific_layer_inputs(
            idx, layer,
            hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict,
            ** kwargs
        )
        if self.gradient_checkpointing and self.is_training:
            layer_outputs = self.base_model._gradient_checkpointing_func(layer.__call__, *input_args, **input_kwargs)
        else:
            layer_outputs = layer(*input_args, **input_kwargs)

        hidden_states, hidden_states_stack, self_attentions_stack, cache = self._update_state(
            layer_outputs,
            hidden_states_stack, self_attentions_stack, cache,
            use_cache, output_attentions, output_hidden_states
        )

        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def _layers_iterator(self, *args, **kwargs) -> Iterable:
        return self.layers

    def _wrapped_forward(
            self, hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
            batch_size, seq_length, prefix_length, device,
            input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
            use_cache, output_attentions, output_hidden_states, return_dict, **kwargs
    ) -> Tuple[
        torch.FloatTensor,
        Optional[Iterable[torch.FloatTensor]],
        Optional[Iterable[torch.FloatTensor]],
        Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]]
    ]:
        for idx, layer in enumerate(self._layers_iterator(**kwargs)):
            hidden_states, hidden_states_stack, self_attentions_stack, cache = self._layer_wrapped_forward(
                idx, layer,
                hidden_states, attention_mask, self_attentions_stack, hidden_states_stack, cache,
                batch_size, seq_length, prefix_length, device,
                input_ids, input_embeddings, position_ids, past_key_values, valid_mask,
                use_cache, output_attentions, output_hidden_states, return_dict,
                **kwargs
            )

        # Apply final normalisation
        hidden_states = self.norm(hidden_states)

        return hidden_states, hidden_states_stack, self_attentions_stack, cache

    def _model_specific_postprocessing(
            self,
            hidden_states: torch.FloatTensor,
            hidden_states_stack: Optional[Iterable[torch.FloatTensor]],
            self_attentions_stack: Optional[Iterable[torch.FloatTensor]],
            cache: Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]],
    ) -> Union[BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions]:
        #
        if isinstance(self.base_model, GPT2Model):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=cache,
                hidden_states=hidden_states_stack,
                attentions=self_attentions_stack
            )
        elif isinstance(self.base_model, LlamaModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=cache,
                hidden_states=hidden_states_stack,
                attentions=self_attentions_stack
            )
        elif isinstance(self.base_model, MistralModel):
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=cache,
                hidden_states=hidden_states_stack,
                attentions=self_attentions_stack
            )
        else:
            raise NotImplementedError(f'Unsupported model type: `{type(self.base_model)}`.')

    def _post_process_output(
            self,
            hidden_states: torch.FloatTensor,
            hidden_states_stack: Optional[Iterable[torch.FloatTensor]],
            self_attentions_stack: Optional[Iterable[torch.FloatTensor]],
            cache: Optional[Union[DynamicCache, Iterable[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]],
            return_dict: Optional[bool],
    ) -> Union[BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        # TODO expand supported models (e.g., GPT-2 uses output with past and cross-attention)
        #
        if cache is not None and isinstance(cache, DynamicCache):
            cache = cache.to_legacy_cache()
        if self_attentions_stack is not None:
            logger.warning(
                'Note: the last tensor in the output `hidden_states` is the non-normalised tensor `last_hidden_state`.'
            )
        #
        if return_dict:
            return self._model_specific_postprocessing(hidden_states, hidden_states_stack, self_attentions_stack, cache)
        else:
            return tuple(v for v in [hidden_states, cache, hidden_states_stack, self_attentions_stack] if v is not None)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs  # As for now, any other additional input is ignored
    ):
        # Prepare the input
        args = self._pre_process_input(
            input_ids, inputs_embeds, position_ids, past_key_values, attention_mask,
            use_cache, output_attentions, output_hidden_states, return_dict
        )
        # Iterate through the wrapped layers
        hidden_states, hidden_states_stack, self_attentions_stack, cache = self._wrapped_forward(*args, **kwargs)

        return self._post_process_output(
            hidden_states, hidden_states_stack, self_attentions_stack, cache, return_dict
        )

        # TODO implement other PreTrainedModel methods


class LMHeadWrapper(ModuleWrapper):
    module_attr = LMHeadAttr
    

class CausalLMWrapper(PreTrainedModelWrapper):
    _auto_model_dtype: Type[PreTrainedModel] = AutoModelForCausalLM
    _wrapper_class: Type[PreTrainedModelWrapper] = PreTrainedModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self._lm_transformer_attr: LMTransformerAttr = self._get_lm_transformer_attr()
        self._wrapper = self._wrapper_class(
            getattr(self._model, self._lm_transformer_attr.value), self._tokenizer
        )

    def _get_lm_transformer_attr(self) -> LMTransformerAttr:
        return _get_module_attr_name(self._model, LMTransformerAttr)

    def save_pretrained(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.base_model.save_pretrained(*args, **kwargs)

    @property
    def base_model(self) -> PreTrainedModel:
        logger.warning('The returned base model may be modified.')
        self.disable_wrapper()
        return self._model

    @property
    def model(self) -> PreTrainedModel:
        return self._wrapper.base_model

    @property
    def lm_head(self) -> nn.Linear:
        return self._model.lm_head

    @property
    def wrapper(self) -> PreTrainedModelWrapper:
        return self._wrapper

    @property
    def is_wrapping(self):
        return isinstance(getattr(self._model, self._lm_transformer_attr.value), self._wrapper_class)

    def enable_wrapper(self):
        if not self.is_wrapping:
            setattr(self._model, self._lm_transformer_attr.value, self._wrapper)

    def disable_wrapper(self):
        if self.is_wrapping:
            setattr(self._model, self._lm_transformer_attr.value, self._wrapper.base_model)

    def forward(self, *args, **kwargs):
        self.enable_wrapper()
        return self._model.forward(*args, **kwargs)

    def _update_wrapper_attributes(self, **kwargs) -> Dict:
        old_attributes = dict()
        for attr, val in kwargs.items():
            old_attributes[attr] = getattr(self.wrapper, attr)
            setattr(self.wrapper, attr, val)
        return old_attributes

    def prepare_inputs_for_generation(self, *args, **kwargs):
        self.enable_wrapper()
        return self._model.prepare_inputs_for_generation(*args, **kwargs)

    # TODO implement other PreTrainedModel methods
