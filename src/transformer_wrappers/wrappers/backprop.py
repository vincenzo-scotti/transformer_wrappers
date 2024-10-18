from functools import partial
import inspect

from typing import Type, Optional, Callable

import torch
import torch.nn.functional as F
from transformers import logging

from .base import ModuleWrapper
from .base import AttentionWrapper, FeedForwardWrapper
from .base import LayerWrapper, LayersWrapper
from .base import PreTrainedModelWrapper, TransformerWrapper, CausalLMWrapper

from .base.constants import *

__all__ = [
    'BackPropTransformerWrapper',
    'BackPropCausalLMWrapper'
]


LOGITS_LAYER: str = 'logits_layer'
LOGITS_FROM_INTERMEDIATE_STATE: str = 'logits_from_intermediate_state'
DETACH_HIDDEN_STATE: str = 'detach_hidden_state'
DETACH_FROM_INTERMEDIATE_STATE: str = 'detach_from_intermediate_state'
BACKPROP_MASK: str = 'backprop_mask'

BACKPROP: str = 'backprop'
GRADIENTS: str = 'gradients'
HIDDEN_STATE: str = 'hidden_state'

logger = logging.get_logger(__name__)


class BackPropAttentionWrapper(AttentionWrapper):
    def get_gradients(self):
        return {
            'qkv_proj': {
                'weights': self.qkv_proj.weight.grad,
                'bias': self.qkv_proj.bias.grad if self.qkv_proj.bias is not None else None
            } if not self.split_proj else {
                f'{proj_id}_proj': {
                    'weights': module.weight.grad,
                    'bias': module.bias.grad if module.bias is not None else None
                } for proj_id, module in zip('qkv', self.qkv_proj)
            },
            'out_proj': {
                'weights': self.out_proj.weight.grad,
                'bias': self.out_proj.bias.grad if self.out_proj.bias is not None else None
            }
        }


class BackPropFeedForwardWrapper(FeedForwardWrapper):
    def get_gradients(self):
        return {
            'up_proj': {
                'weights': self.up_proj.weight.grad,
                'bias': self.up_proj.bias.grad if self.up_proj.bias is not None else None
            },
            'down_proj': {
                'weights': self.down_proj.weight.grad,
                'bias': self.down_proj.bias.grad if self.down_proj.bias is not None else None
            }
        } | ({'gate_proj': {
            'weights': self.gate_proj.weight.grad,
            'bias': self.gate_proj.bias.grad if self.gate_proj.grad is not None else None
        }} if self.gate_proj is not None else dict())


class BackPropLayerWrapper(LayerWrapper):
    module_output: str = 'parallel_layer_output'

    _attention_dtype: Type[ModuleWrapper] = BackPropAttentionWrapper
    _feed_forward_dtype: Type[ModuleWrapper] = BackPropFeedForwardWrapper

    # TODO fixme
    def _attn_forward(
            self,
            *args,
            current_hidden_state: Optional[torch.FloatTensor],
            add_attn_residual: bool = True,
            detach_attention: bool = False,
            backprop_mask: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Initial Normalisation
        current_hidden_state = self.initial_norm.forward(current_hidden_state)
        if detach_attention:
            tmp_hidden_state = current_hidden_state.detach()
            tmp_hidden_state[backprop_mask] = current_hidden_state[backprop_mask]
            current_hidden_state = tmp_hidden_state
        # Self attention
        attention_output = self.attention_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.attention_wrapper.module_output)
        if add_attn_residual:
            current_hidden_state = attention_output[self.attention_wrapper.module_output] + residual
        else:
            current_hidden_state = attention_output[self.attention_wrapper.module_output]
        if detach_attention:
            tmp_hidden_state = current_hidden_state.detach()
            tmp_hidden_state[backprop_mask] = current_hidden_state[backprop_mask]
            current_hidden_state = tmp_hidden_state
        #
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            self.intermediate_module_output: current_hidden_state,
            ADD_ATTN_RESIDUAL: add_attn_residual,
            self.attention_wrapper.module_output: attention_output,
            BACKPROP_MASK: backprop_mask
        }

        return output

    # TODO fixme
    def _ffnn_forward(
            self,
            current_hidden_state,
            add_ffnn_residual: bool = True,
            detach_feed_forward: bool = False,
            backprop_mask: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Intermediate Normalisation
        current_hidden_state = self.intermediate_norm.forward(current_hidden_state)
        if detach_feed_forward:
            tmp_hidden_state = current_hidden_state.detach()
            tmp_hidden_state[backprop_mask] = current_hidden_state[backprop_mask]
            current_hidden_state = tmp_hidden_state
        # Feed-Forward
        ffnn_output = self.feed_forward_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.feed_forward_wrapper.module_output)
        if add_ffnn_residual:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output] + residual  # TODO verify this
        else:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output]
        if detach_feed_forward:
            tmp_hidden_state = current_hidden_state.detach()
            tmp_hidden_state[backprop_mask] = current_hidden_state[backprop_mask]
            current_hidden_state = tmp_hidden_state
        # Extend input with module output
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_FFNN_RESIDUAL: add_ffnn_residual,
            self.feed_forward_wrapper.module_output: ffnn_output,
            BACKPROP_MASK: backprop_mask
        }

        return output

    def get_gradients(self):
        return {
            'attention': self.attention_wrapper.get_gradients(),
            'ffnn': self.feed_forward_wrapper.get_gradients()
        }


class BackPropLayersWrapper(LayersWrapper):
    TMP_HIDDEN_STATE: str = 'tmp_hidden_state'

    _layer_dtype: Type[ModuleWrapper] = BackPropLayerWrapper

    def _wrapped_forward(
            self,
            detach_hidden_state: Optional[int] = None,
            detach_from_intermediate_state: bool = False,
            **kwargs
    ):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            output = layer_wrapper.forward(
                layer_idx=layer_idx,
                detach_attention=(
                        (layer_idx == detach_hidden_state and detach_from_intermediate_state) or
                        layer_idx > detach_hidden_state
                ) if detach_hidden_state is not None else False,
                detach_feed_forward=layer_idx >= detach_hidden_state if detach_hidden_state is not None else False,
                **output
            )
            # Update model state
            output = self._update_state(**output)

        output |= {
            DETACH_HIDDEN_STATE: detach_hidden_state,
            DETACH_FROM_INTERMEDIATE_STATE: detach_from_intermediate_state
        }

        return output

    def get_gradients(self):
        return [layer.get_gradients() for layer in self.layer_wrappers]


class BackPropTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[ModuleWrapper] = BackPropLayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.logits_layer: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(LOGITS_LAYER)
        self.logits_from_intermediate_state: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(
            LOGITS_FROM_INTERMEDIATE_STATE, False
        )
        self.detach_hidden_state: Optional[int] = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(
            DETACH_HIDDEN_STATE
        )
        self.detach_from_intermediate_state: bool = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(
            DETACH_FROM_INTERMEDIATE_STATE, False
        )

    @property
    def wrapper_args(self):
        return super().wrapper_args | {
            LOGITS_LAYER,
            LOGITS_FROM_INTERMEDIATE_STATE,
            DETACH_HIDDEN_STATE,
            DETACH_FROM_INTERMEDIATE_STATE
        }

    def _pre_process_input(
        self,
        *args,
        logits_layer: Optional[int] = None,
        logits_from_intermediate_state: bool = False,
        detach_hidden_state: Optional[int] = None,
        detach_from_intermediate_state: bool = False,
        **kwargs
    ):
        kwargs = super()._pre_process_input(*args, **kwargs)
        #
        logits_layer = logits_layer if logits_layer is not None else self.logits_layer
        logits_from_intermediate_state = logits_from_intermediate_state if logits_from_intermediate_state is not None else self.logits_from_intermediate_state
        detach_hidden_state = detach_hidden_state if detach_hidden_state is not None else self.detach_hidden_state
        detach_from_intermediate_state = detach_from_intermediate_state if detach_from_intermediate_state is not None else self.detach_from_intermediate_state
        #
        backprop_mask = kwargs[LABELS] != -100 if LABELS in kwargs else None
        #
        kwargs |= {
            LOGITS_LAYER: logits_layer,
            LOGITS_FROM_INTERMEDIATE_STATE: logits_from_intermediate_state,
            DETACH_HIDDEN_STATE: detach_hidden_state,
            DETACH_FROM_INTERMEDIATE_STATE: detach_from_intermediate_state,
            BACKPROP_MASK: backprop_mask
        }

        return kwargs

    def _wrapped_forward(
            self, detach_hidden_state: Optional[int] = None, backprop_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        #
        output = super()._wrapped_forward(
            detach_hidden_state=detach_hidden_state, backprop_mask=backprop_mask, **kwargs
        )
        #
        if detach_hidden_state is not None:
            tmp_hidden_state = output[self.model_output].detach()
            tmp_hidden_state[backprop_mask] = output[self.model_output][backprop_mask]
            output[self.model_output] = tmp_hidden_state

        return output


class BackPropCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = BackPropTransformerWrapper

    @torch.no_grad()
    def get_interpolated_decoder(self, layer: Optional[int] = None) -> Callable:
        # Zero-based indexing starting from input embeddings
        if layer is None:
            layer = len(self.transformer.layers)
        #
        if layer == 0:
            w = self.get_input_embeddings().weight
        elif layer == len(self.transformer_wrapper.layers_wrapper.layer_wrappers):
            w = self.lm_head_wrapper.base_module.weight
        else:
            w = (
                ((1 - layer / len(self.transformer_wrapper.layers_wrapper.layer_wrappers)) * self.get_input_embeddings().weight) +
                ((layer / len(self.transformer_wrapper.layers_wrapper.layer_wrappers)) * self.lm_head_wrapper.base_module.weight)
            )

        return partial(F.linear, weight=w)

    def _post_process_output(
            self,
            base_model_output: bool = False,
            labels: Optional[torch.LongTensor] = None,
            logits_layer: Optional[int] = None,
            logits_from_intermediate_state: bool = False,
            **kwargs
    ):
        kwargs = super()._post_process_output(
            base_model_output=base_model_output,
            labels=labels,
            logits_layer=logits_layer,
            logits_from_intermediate_state=logits_from_intermediate_state,
            **kwargs
        )
        #
        if not base_model_output and labels is not None:
            if logits_layer is None:
                hidden_state = kwargs[OUT_HIDDEN_STATE]
                logits = kwargs[LOGITS]
                loss = kwargs[LOSS]
            else:
                self.zero_grad()
                hidden_state = kwargs[
                    HIDDEN_STATES if not logits_from_intermediate_state else INTERMEDIATE_HIDDEN_STATES
                ][logits_layer - int(logits_from_intermediate_state)]
                # TODO add normalisation for intermediate layers
                logits = self.get_interpolated_decoder(layer=logits_layer)(hidden_state)
                loss = self._loss(logits, labels) if labels is not None else None
            #
            loss.backward()
            #
            kwargs |= {
                BACKPROP: {
                    HIDDEN_STATE: hidden_state,
                    LOGITS: logits,
                    LOSS: loss,
                    GRADIENTS: self.transformer_wrapper.layers_wrapper.get_gradients()
                }
            }

        return kwargs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        if not self.is_wrapping:
            return self.base_model.generate(*args, **kwargs)

        generate_output = super(PreTrainedModelWrapper, self).generate(*args, **kwargs)
        # Re-run through layers to collect all data  # TODO find better solution
        if return_inner_states or not self.is_benchmarking:
            #
            labels = generate_output.clone()
            labels[:, :(kwargs.get(INPUT_IDS, kwargs.get(INPUT_EMBEDS))).size(1)] = -100
            #
            return self.forward(
                input_ids=generate_output,
                **{
                    k: kwargs.get(k) for k in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())
                    if k not in {'args', 'kwargs', 'self', 'base_model_output'}
                },
                labels=labels,
                return_dict=True,
                output_attentions=True,
                use_cache=True,
                output_hidden_states=True,
                return_attention_output=True,  # Self-attention layer output
                return_feed_forward_output=True,
                return_intermediate_hidden_states=True
            ) | {OUTPUT_IDS: generate_output}
        else:
            return generate_output

    def prepare_inputs_for_generation(
            self,
            *args,
            logits_layer: Optional[int] = None,
            logits_from_intermediate_state: bool = False,
            detach_hidden_state: Optional[int] = None,
            detach_from_intermediate_state: bool = False,
            **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs) | {
            LOGITS_LAYER: logits_layer,
            LOGITS_FROM_INTERMEDIATE_STATE: logits_from_intermediate_state,
            DETACH_HIDDEN_STATE: detach_hidden_state,
            DETACH_FROM_INTERMEDIATE_STATE: detach_from_intermediate_state
        }

        return inputs
