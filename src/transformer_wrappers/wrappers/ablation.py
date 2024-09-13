from typing import Type, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, replace

import torch

from .base import TransformerWrapper
from .base import LayerWrapper, LayersWrapper, ModuleWrapper
from .parameter import ParameterCausalLMWrapper
from .constants import * # pylint:disable=W0401,W0614


class AblationPosition(Enum):
    ATTENTION = "ablation_attention"
    FFNN = "ablation_ffnn"

@dataclass
class AblationInfo:
    layers: Tuple[int]
    token: int
    position: AblationPosition
    index: int = -1


class AblationLayerWrapper(LayerWrapper):
    ABLATION_CANDIDATES = "abl_cand"

    def _wrapped_forward_ablation(
        self,
        abl_cand = [],
        skip_attention: bool = False,
        skip_ffnn: bool = False,
        **kwargs
    ):
        output = kwargs
        if AblationPosition.ATTENTION in [c.position for c in abl_cand]:
            output[self.attention_wrapper.module_output] = {
                self.attention_wrapper.module_output: None,
                self.attention_wrapper.attention_weights: None,
                self.attention_wrapper.key_value: None
            }
            output[self.intermediate_module_output] = output[CURR_HIDDEN_STATE]
            output["past_key_values"].update(
                # TODO: problematic to skip layer 0
                output["past_key_values"][-1][0],
                output["past_key_values"][-1][1],
                output["layer_idx"]
            )
        else:
            output = self._attn_forward(**output)
        if AblationPosition.FFNN in [c.position for c in abl_cand]:
            output[self.feed_forward_wrapper.module_output] = {
                self.feed_forward_wrapper.module_output: None
            }
        else:
            output = self._ffnn_forward(**output)
        #
        output |= {self.module_output: output[CURR_HIDDEN_STATE]}

        return output

    def _wrapped_forward(self, **kwargs): # pylint:disable=W0221
        return self._wrapped_forward_ablation(
            **kwargs) if AblationLayerWrapper.ABLATION_CANDIDATES in kwargs else super()._wrapped_forward(
            self, **kwargs)


class AblationLayersWrapper(LayersWrapper):
    _layer_dtype: Type[ModuleWrapper] = AblationLayerWrapper

    def _wrapped_forward(self, **kwargs):
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            output = self._ablation_candiate_gen(layer_idx=layer_idx, **output)
            output = layer_wrapper.forward(layer_idx=layer_idx, **output)
            # Update model state
            output = self._update_state(**output)

        return output

    def _ablation_candiate_gen(self, layer_idx, **kwargs):
        ablations = kwargs[AblationCausalLMWrapper.ABLATIONS_PARAMETER] if AblationCausalLMWrapper.ABLATIONS_PARAMETER in kwargs else []
        position_ids = kwargs[POSITION_IDS].squeeze()
        if ablations and (
            abl_candidates := [i for i in ablations if (
                i.layers[0] <= layer_idx and layer_idx <= i.layers[1]) and 
                i.token in position_ids
            ]
        ):
            for abl in abl_candidates:
                abl.index = (position_ids == abl.token).nonzero(as_tuple=True)[0]
            kwargs |= {AblationLayerWrapper.ABLATION_CANDIDATES: abl_candidates}
        return kwargs


class AblationTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[TransformerWrapper] = AblationLayersWrapper


class AblationCausalLMWrapper(ParameterCausalLMWrapper):

    ABLATIONS_PARAMETER = "ablation_info"

    _transformer_dtype: Type[TransformerWrapper] = AblationTransformerWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameters([AblationCausalLMWrapper.ABLATIONS_PARAMETER])

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        # Adjust layer ranges
        ablations = kwargs.pop(AblationCausalLMWrapper.ABLATIONS_PARAMETER, [])
        new_ablations = []
        for ab in ablations:
            if type(ab.layers) not in [list, tuple]:
                ab.layers = (ab.layers, ab.layers)
        kwargs |= {AblationCausalLMWrapper.ABLATIONS_PARAMETER: ablations}

        return ParameterCausalLMWrapper.generate(self, *args, **kwargs)
