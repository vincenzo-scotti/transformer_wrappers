from typing import Type, Optional, List, Dict, Callable
from enum import Enum
from dataclasses import dataclass

import torch

from .base import TransformerWrapper
from .base import LayerWrapper, LayersWrapper, ModuleWrapper
from .parameter import ParameterCausalLMWrapper
from .ablation import AblationCausalLMWrapper, AblationTransformerWrapper, AblationLayersWrapper, AblationLayerWrapper, AblationPosition
from .constants import * # pylint:disable=W0401,W0614


class InjectionStrategy(str, Enum):
    REPLACE = "replace"
    REMOVE_FIRST_COMPONENT = "remove_fc"

class InjectPosition(str, Enum):
    ATTENTION = "inject_attention"
    INTERMEDIATE = "inject_intermediate"
    FFNN = "inject_ffnn"
    OUTPUT = "inject_output"

@dataclass
class InjectInfo:
    layer: int
    token: int
    position: InjectPosition
    embedding: torch.FloatTensor
    strategy: InjectionStrategy = InjectionStrategy.REPLACE
    decoding_matrix: torch.Tensor = None
    decoding_norm: Callable = lambda x: x
    index: int = -1


def _inject_replace(hidden_states, inject_info: InjectInfo):
    inj_emb = inject_info.embedding
    norm_inj_emb = inject_info.decoding_norm(inj_emb)
    hidden_states[0, inject_info.index, :] = norm_inj_emb
    return hidden_states

def _inject_remove_fc(hidden_states, inject_info: InjectInfo):
    inj_emb = inject_info.embedding
    decoding_matrix = inject_info.decoding_matrix.to(hidden_states.device)

    inj_emb = torch.nn.functional.normalize(inj_emb, p=2, dim=-1)

    target_state = hidden_states[0, inject_info.index, :]
    target_state_norm = inject_info.decoding_norm(target_state)
    target_fc_id = torch.argmax(torch.matmul(target_state_norm, decoding_matrix.T))
    target_fc = decoding_matrix[target_fc_id]
    target_fc = torch.nn.functional.normalize(target_fc, p=2, dim=-1)

    scaling_factor = torch.dot(target_state.squeeze(), target_fc.squeeze())
    inject = target_state + scaling_factor * (inj_emb - target_fc)

    hidden_states[0, inject_info.index, :] = inject

    return hidden_states

_INJECTION_STRATEGY_MAP = {
    InjectionStrategy.REPLACE: _inject_replace,
    InjectionStrategy.REMOVE_FIRST_COMPONENT: _inject_remove_fc,
}

def _inject(
    hidden_states: torch.FloatTensor,
    inject_info: InjectInfo,
):
    return _INJECTION_STRATEGY_MAP[inject_info.strategy](hidden_states, inject_info)


class InjectLayerWrapper(LayerWrapper):
    INJECT_CANDIDATES = "inj_cand"

    def _attn_forward_inject(
            self, current_hidden_state: Optional[torch.FloatTensor],
            add_attn_residual: bool = True, inj_cand: InjectInfo = None, **kwargs):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Initial Normalisation
        current_hidden_state = self.initial_norm.forward(current_hidden_state)
        # Self attention
        attention_output = self.attention_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.attention_wrapper.module_output)

        # Injection ATTENTION
        attention_output |= {
            self.attention_wrapper.module_output: _inject(
                attention_output[self.attention_wrapper.module_output], inj_info
            )
            for inj_info in inj_cand if inj_info.position == InjectPosition.ATTENTION
        }

        if add_attn_residual:
            current_hidden_state = attention_output[self.attention_wrapper.module_output] + residual
        else:
            current_hidden_state = attention_output[self.attention_wrapper.module_output]

        # Injection INTERMEDIATE
        current_hidden_state = [
            _inject(current_hidden_state, inj_info)
            if inj_info.position == InjectPosition.INTERMEDIATE else current_hidden_state
            for inj_info in inj_cand
        ][0]

        #
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            self.intermediate_module_output: current_hidden_state,
            ADD_ATTN_RESIDUAL: add_attn_residual,
            self.attention_wrapper.module_output: attention_output,
            InjectLayerWrapper.INJECT_CANDIDATES: inj_cand
        }

        return output

    def _ffnn_forward_inject(
            self, current_hidden_state: Optional[torch.FloatTensor],
            add_ffnn_residual: bool = True, inj_cand: InjectInfo = None, **kwargs):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Intermediate Normalisation
        current_hidden_state = self.intermediate_norm.forward(current_hidden_state)
        # Feed-Forward
        ffnn_output = self.feed_forward_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.feed_forward_wrapper.module_output)

        # Injection FFNN
        ffnn_output |= {
            self.feed_forward_wrapper.module_output: _inject(
                ffnn_output[self.feed_forward_wrapper.module_output], inj_info
            )
            for inj_info in inj_cand if inj_info.position == InjectPosition.FFNN
        }

        if add_ffnn_residual:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output] + residual  # TODO verify this
        else:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output]

        # Injection OUTPUT
        current_hidden_state = [
            _inject(current_hidden_state, inj_info)
            if inj_info.position == InjectPosition.OUTPUT else current_hidden_state
            for inj_info in inj_cand
        ][0]

        # Extend input with module output
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_FFNN_RESIDUAL: add_ffnn_residual,
            self.feed_forward_wrapper.module_output: ffnn_output
        }

        return output

    def _wrapped_forward_inject(
        self,
        skip_attention: bool = False,
        skip_ffnn: bool = False,
        **kwargs
    ):
        output = kwargs
        output = self._attn_forward_inject(**output)
        output = self._ffnn_forward_inject(**output)
        #
        output |= {self.module_output: output[CURR_HIDDEN_STATE]}

        return output

    def _wrapped_forward(self, **kwargs): # pylint:disable=W0221
        return self._wrapped_forward_inject(
            **kwargs) if InjectLayerWrapper.INJECT_CANDIDATES in kwargs else super()._wrapped_forward(
            self, **kwargs)


class InjectLayersWrapper(LayersWrapper):
    _layer_dtype: Type[ModuleWrapper] = InjectLayerWrapper

    def _wrapped_forward(self, **kwargs):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            output = self._injections_candiate_gen(layer_idx=layer_idx, **output)
            output = layer_wrapper.forward(layer_idx=layer_idx, **output)
            # Update model state
            output = self._update_state(**output)

        return output

    def _injections_candiate_gen(self, layer_idx, **kwargs):
        injections = kwargs[InjectCausalLMWrapper.INJECTS_PARAMETER] if InjectCausalLMWrapper.INJECTS_PARAMETER in kwargs else []
        position_ids = kwargs[POSITION_IDS].squeeze()
        if injections and (
            inj_candidates := [
                i for i in injections if i.layer == layer_idx and i.token in position_ids
            ]
        ):
            for inj in inj_candidates:
                inj.index = (position_ids == inj.token).nonzero(as_tuple=True)[0]
            kwargs |= {InjectLayerWrapper.INJECT_CANDIDATES: inj_candidates}
        return kwargs


class InjectTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[TransformerWrapper] = InjectLayersWrapper


class InjectCausalLMWrapper(ParameterCausalLMWrapper):

    INJECTS_PARAMETER = "inject_info"

    _transformer_dtype: Type[TransformerWrapper] = InjectTransformerWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameters([InjectCausalLMWrapper.INJECTS_PARAMETER])

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        return ParameterCausalLMWrapper.generate(self, *args, **kwargs)

    def prepare_injections(self, injections: List[Dict], layer_offset: bool = True):
        layer_offset = 1 if layer_offset else 0
        return [
            InjectInfo(
                **inj | {
                    # Add layer visualization offset
                    "layer": inj["layer"] - layer_offset,
                }
            )
            for inj in injections
        ]




class AblInjLayerWrapper(InjectLayerWrapper, AblationLayerWrapper):

    FORWARD_CHECK = lambda keys, map: sum([[key] if key in map else [] for key in keys], [])
    FORWARD_MAP = {
        frozenset([InjectLayerWrapper.INJECT_CANDIDATES, AblationLayerWrapper.ABLATION_CANDIDATES]): 
            lambda self, kwargs: self._wrapped_forward_ablinj(**kwargs),
        frozenset([InjectLayerWrapper.INJECT_CANDIDATES]):
            lambda self, kwargs: InjectLayerWrapper._wrapped_forward(self, **kwargs),
        frozenset([AblationLayerWrapper.ABLATION_CANDIDATES]):
            lambda self, kwargs: AblationLayerWrapper._wrapped_forward(self, **kwargs),
        frozenset([]):
            lambda self, kwargs: LayerWrapper._wrapped_forward(self, **kwargs),
    }

    def _wrapped_forward_ablinj(
        self,
        abl_cand = [],
        skip_attention: bool = False,
        skip_ffnn: bool = False,
        **kwargs
    ):
        output = kwargs
        if AblationPosition.ATTENTION in [c.position for c in abl_cand]:
            residual_without_attention = output[CURR_HIDDEN_STATE]
            output = self._attn_forward_inject(**output)
            output[self.intermediate_module_output] = residual_without_attention
            output[CURR_HIDDEN_STATE] = residual_without_attention
        else:
            output = self._attn_forward_inject(**output)

        if AblationPosition.FFNN in [c.position for c in abl_cand]:
            current_output = output[CURR_HIDDEN_STATE]
            output = self._ffnn_forward_inject(**output)
            output[CURR_HIDDEN_STATE] = current_output
        else:
            output = self._ffnn_forward_inject(**output)

        output |= {self.module_output: output[CURR_HIDDEN_STATE]}
        
        return output

    def _wrapped_forward(self, **kwargs): # pylint:disable=W0221
        return AblInjLayerWrapper.FORWARD_MAP[frozenset(AblInjLayerWrapper.FORWARD_CHECK(
            [InjectLayerWrapper.INJECT_CANDIDATES, AblationLayerWrapper.ABLATION_CANDIDATES], 
            kwargs,
        ))](self, kwargs)

class AblInjLayersWrapper(InjectLayersWrapper, AblationLayersWrapper):
    _layer_dtype: Type[ModuleWrapper] = AblInjLayerWrapper

    def _wrapped_forward(self, **kwargs):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            output = self._injections_candiate_gen(layer_idx=layer_idx, **output)
            output = self._ablation_candiate_gen(layer_idx=layer_idx, **output)
            output = layer_wrapper.forward(layer_idx=layer_idx, **output)
            # Update model state
            output = self._update_state(**output)

        return output

class AblInjTransformerWrapper(InjectTransformerWrapper, AblationTransformerWrapper):
    _layers_dtype: Type[TransformerWrapper] = AblInjLayersWrapper

class AblInjCausalLMWrapper(InjectCausalLMWrapper, AblationCausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = AblInjTransformerWrapper

    def __init__(self, *args, **kwargs):
        ParameterCausalLMWrapper.__init__(self, *args, **kwargs)
        self.add_parameters([InjectCausalLMWrapper.INJECTS_PARAMETER, AblationCausalLMWrapper.ABLATIONS_PARAMETER])

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        return AblationCausalLMWrapper.generate(self, *args, **kwargs)