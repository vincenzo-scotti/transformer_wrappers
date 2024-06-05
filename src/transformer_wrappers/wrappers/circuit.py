import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GemmaPreTrainedModel, GPT2PreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from transformers import logging



from .constants import *

from .base import FeedForwardWrapper, LayerWrapper, LayersWrapper, TransformerWrapper, \
    CausalLMWrapper, ModuleWrapper

from enum import Enum
from typing import Union, Optional, Type, Tuple, Dict, List, Iterable

# TODO fix base model/module properties and wrapper enable/disable methods
# TODO implement gradient checkpointing
# TODO implement training with adapters
# TODO test past key-value


__all__ = [
    'SHARED_STRUCTURE_MODELS',
    'SHARED_STRUCTURE_LAYERS',
    'ModuleWrapper',
    'PreTrainedModelWrapper',
    'EmbeddingWrapper',
    'AttentionWrapper',
    'FeedForwardWrapper',
    'LayerWrapper',
    'LayersWrapper',
    'LMHeadWrapper',
    'TransformerWrapper',
    'CausalLMWrapper'
]

logger = logging.get_logger(__name__)

SHARED_STRUCTURE_MODELS = (GemmaPreTrainedModel, LlamaPreTrainedModel, MistralPreTrainedModel)
SHARED_STRUCTURE_LAYERS = (GemmaDecoderLayer, LlamaDecoderLayer, MistralDecoderLayer)


class TransformerEmbeddingAttr(Enum):
    EMBED_TOKENS = 'embed_tokens'
    WTE = 'wte'


class TransformerPositionEmbeddingAttr(Enum):
    WPE = 'wpe'


class FFNNUpProjectionAttr(Enum):
    UP_PROJ = 'up_proj'
    C_FC = 'c_fc'


class FFNNGateProjectionAttr(Enum):
    GATE_PROJ = 'gate_proj'


class FFNNActivationFunctionAttr(Enum):
    ACT_FN = 'act_fn'
    ACT = 'act'


class FFNNDownProjectionAttr(Enum):
    DOWN_PROJ = 'down_proj'
    C_PROJ = 'c_proj'


class FFNNDropoutAttr(Enum):
    DROPOUT = 'dropout'


class LayerInitialNormAttr(Enum):
    INPUT_LAYERNORM = 'input_layernorm'
    LN_1 = 'ln_1'


class LayerAttentionAttr(Enum):
    SELF_ATTN = 'self_attn'
    ATTN = 'attn'


class LayerIntermediateNormAttr(Enum):
    INPUT_LAYERNORM = 'post_attention_layernorm'
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
    FFNNGateProjectionAttr, FFNNUpProjectionAttr, FFNNDownProjectionAttr, FFNNActivationFunctionAttr, FFNNDropoutAttr,
    LayerInitialNormAttr, LayerAttentionAttr, LayerIntermediateNormAttr, LayerFeedForwardAttr,
    TransformerEmbeddingAttr, TransformerPositionEmbeddingAttr, TransformerLayersAttr, TransformerNormAttr,
    LMTransformerAttr, LMHeadAttr
]


def _get_module_attr_name(model: nn.Module, attr_names: Type[AttrEnumTypes]):
    #
    for attr in attr_names:
        if hasattr(model, attr.value):
            return attr
    #
    raise ValueError(f'Unsupported module type `{type(model)}` for attribute `{attr_names}`.')


OVERWRITE_ACTIVATIONS: str = "overwrite_activations"


class FFCircuitType(Enum):
    GATE = "gate"
    UP = "up"
    KEY = "key"
    OUT = "out"


class CircuitFeedForwardWrapper(FeedForwardWrapper):
    _module_name: str = 'circuit_feed_forward_module'

    def _wrapped_forward(self, current_hidden_state: Optional[torch.FloatTensor],
                         overwrite_activation: Optional[Dict] = None, **kwargs):
        """
        Overwrite activations is a dictionary in the form {"component_name": {"indices":List, "values": List}}
        to be overwrite
        """
        if current_hidden_state is None:
            raise ValueError()
        #
        # pop components by name
        up_proj = overwrite_activation.get(FFCircuitType.UP.value)
        down_proj = overwrite_activation.get(FFCircuitType.OUT.value)
        gate_act_proj = overwrite_activation.get(FFCircuitType.GATE.value)
        inner_act = overwrite_activation.get(FFCircuitType.KEY.value)

        if current_hidden_state is None:
            raise ValueError()

        fine_grained_output = any((
            kwargs[RETURN_FFNN_INNER_ACTIVATIONS],
            kwargs[RETURN_FFNN_UP_PROJ_OUTPUT],
            kwargs[RETURN_FFNN_GATE_OUTPUT]
        ))
        #
        if not fine_grained_output:
            up_proj_output = gate_output = inner_activations = None
            ffnn_output = self._module.forward(current_hidden_state)
        elif isinstance(self.super_wrapper.base_module, GPT2Block):
            up_proj_output = self.up_proj(current_hidden_state)
            gate_output = None
            inner_activations = self.act_fn(up_proj_output)
            ffnn_output = self.down_proj(inner_activations)
            ffnn_output = self.dropout(ffnn_output)
        elif (
                isinstance(self.super_wrapper.base_module, (MistralDecoderLayer, GemmaDecoderLayer)) or
                (isinstance(self.super_wrapper.base_module,
                            LlamaDecoderLayer) and self._module.config.pretraining_tp <= 1)
        ):
            up_proj_output = self.up_proj(current_hidden_state)
            # Force up activations
            if up_proj is not None:
                up_proj_output[..., -1, up_proj['indices']] = up_proj['values']

            gate_output = self.act_fn(self.gate_proj(current_hidden_state))
            # Force up activations
            if gate_act_proj is not None:
                gate_output[..., -1, gate_act_proj['indices']] = torch.tensor(gate_act_proj['values'],
                                                                              dtype=gate_output.dtype,
                                                                              device=gate_output.device)

            inner_activations = gate_output * up_proj_output
            if inner_act is not None:
                inner_activations[..., -1, inner_act['indices']] = inner_act['values']

            ffnn_output = self.down_proj(inner_activations)
            if down_proj is not None:
                ffnn_output[..., -1, down_proj['indices']] = down_proj['values']

        elif isinstance(self.super_wrapper.base_module, LlamaDecoderLayer) and self._module.config.pretraining_tp > 1:
            # Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L200
            slice = self._module.intermediate_size // self._module.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_output = torch.cat([
                F.linear(current_hidden_state, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ], dim=-1)
            gate_output = self.act_fn(gate_output)
            up_proj_output = torch.cat([
                F.linear(current_hidden_state, up_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ], dim=-1)

            inner_activations = (gate_output * up_proj_output).split(slice, dim=2)
            ffnn_output = [
                F.linear(inner_activations[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            ffnn_output = sum(ffnn_output)
        else:
            raise NotImplementedError(f'Unsupported layer type: `{type(self.super_wrapper.base_module)}`.')

        #
        output = kwargs | {
            self.module_output: ffnn_output,
            self.feed_forward_up_proj_output: up_proj_output,
            self.feed_forward_gate_output: gate_output,
            self.feed_forward_inner_activations: inner_activations,
            CURR_HIDDEN_STATE: current_hidden_state
        }

        return output


class CircuitLayerWrapper(LayerWrapper):
    # module_output: str = 'circuit_layer_output'
    _feed_forward_dtype: Type[ModuleWrapper] = CircuitFeedForwardWrapper


class CircuitLayersWrapper(LayersWrapper):
    _layer_dtype: Type[ModuleWrapper] = CircuitLayerWrapper

    def get_layer_wrappers_iterator(self, overwrite_activations: List[Dict]) -> Iterable:
        return zip(super().get_layer_wrappers_iterator(), overwrite_activations)

    def _update_state(self, overwrite_activation: Optional[Dict] = None, **kwargs):
        return super()._update_state(**kwargs)

    def _wrapped_forward(self, overwrite_activations: Optional[List[Dict]] = None, **kwargs):
        # TODO definire default per overwrite_activations
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, (layer_wrapper, overwrite_activation) in enumerate(
                self.get_layer_wrappers_iterator(overwrite_activations)):
            # Apply layer transformation
            output = layer_wrapper.forward(layer_idx=layer_idx, overwrite_activation=overwrite_activation, **output)
            # Update model state
            output = self._update_state(**output)

        return output | {OVERWRITE_ACTIVATIONS: overwrite_activations}


class CirctuitTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[ModuleWrapper] = CircuitLayersWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _pre_process_input(
            self,
            *args,
            overwrite_activations: Optional[Dict] = None,
            **kwargs
    ):
        kwargs = super()._pre_process_input(*args, **kwargs)
        # Define here the dafault static parameter shape/value
        # Padding of missing layers: getting a dict with {layer: overwrite_activations}
        # Returning a list of n_layers overwrite activations
        if overwrite_activations is None:
            overwrite_activations = {}

        overwrite_activations_padded = [overwrite_activations.get(layer, {}) for layer in range(len(self.layers))]
        kwargs |= {
            OVERWRITE_ACTIVATIONS: overwrite_activations_padded
        }

        return kwargs


class CircuitCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = CirctuitTransformerWrapper

    def compute_addition(
            self,
            input_str: str,
            max_length: int,
            do_sample: bool,
            *args,
            overwrite_activations: Optional[Dict] = None,
            **kwargs
    ):
        input_encodings = self.tokenizer(input_str,
                                         return_tensors='pt'
                                         ).to(self._model.device)
        first_output_digit_index = len(self.tokenizer.convert_ids_to_tokens(
            input_encodings['input_ids'].squeeze()))

        if overwrite_activations:
            # Force value to activations
            # Model forward for input encoding
            # self.enable_wrapper()
            output_forward = self.forward(
                **input_encodings,
                return_dict=True,
                output_attentions=False,
                use_cache=True,
                output_hidden_states=False,
                return_attention_output=False,  # Self-attention layer output
                return_feed_forward_output=False,
                return_feed_forward_inner_activations=False,
                return_feed_forward_up_proj_output=False,
                return_feed_forward_gate_output=True,
                overwrite_activations=overwrite_activations
            )
            input_ids = torch.cat(
                [input_encodings['input_ids'], torch.argmax(output_forward['logits'][0, -1]).reshape(1, 1)], dim=1)

            # Generate with past key values
            # self.enable_benchmarking()
            self.disable_wrapper()
            generated_output = self.generate(
                input_ids,
                past_key_values=output_forward["past_key_values"],
                max_length=16,
                do_sample=False
            )
            self.enable_wrapper()

            output_str = self.tokenizer.decode(generated_output['output_ids'].squeeze(),
                                               return_tensors='pt'
                                               )
            result_substring = output_str.split('=')[1].split("\n")[0]

            return result_substring
        else:
            output = self.model.generate(input_encodings.input_ids,
                                         do_sample=do_sample,
                                         max_length=max_length)
            output_str = self.tokenizer.decode(output['output_ids'].squeeze())
            result_substring = output_str.split('=')[1].split("\n")[0]
            result = int("".join([ele for ele in result_substring if ele.isdigit()]))
