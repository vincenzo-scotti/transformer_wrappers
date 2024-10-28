from enum import Enum
from typing import Type, Union


__all__ = [
    'TransformerEmbeddingAttr',
    'TransformerPositionEmbeddingAttr',
    'AttnQKVProjectionAttr',
    'AttnQKVProjectionsAttr',
    'AttnOutProjectionAttr',
    'AttnDropoutAttr',
    'FFNNUpProjectionAttr',
    'FFNNGateProjectionAttr',
    'FFNNActivationFunctionAttr',
    'FFNNDownProjectionAttr',
    'FFNNDropoutAttr',
    'LayerInitialNormAttr',
    'LayerAttentionAttr',
    'LayerAttentionDropoutAttr',
    'LayerIntermediateNormAttr',
    'LayerFeedForwardAttr',
    'LayerFeedForwardDropoutAttr'
    'TransformerLayersAttr',
    'TransformerNormAttr',
    'LMTransformerAttr',
    'LMHeadAttr',
    'AttrEnumTypes',
    'MultiAttrEnumTypes'
]


class TransformerEmbeddingAttr(Enum):
    EMBED_TOKENS = 'embed_tokens'
    WTE = 'wte'
    EMBED_IN = 'embed_in'


class TransformerPositionEmbeddingAttr(Enum):
    WPE = 'wpe'


class AttnQKVProjectionAttr(Enum):
    C_ATTN = 'c_attn'
    QUERY_KEY_VALUE = 'query_key_value'


class AttnQKVProjectionsAttr(Enum):
    QKV_ATTN = ('q_proj', 'k_proj', 'v_proj')


class AttnOutProjectionAttr(Enum):
    C_PROJ = 'c_proj'
    O_PROJ = 'o_proj'
    DENSE = 'dense'


class AttnDropoutAttr(Enum):
    DROPOUT = 'dropout'
    ATTENTION_DROPOUT = 'attention_dropout'


class FFNNUpProjectionAttr(Enum):
    UP_PROJ = 'up_proj'
    C_FC = 'c_fc'
    DENSE_H_TO_4H = 'dense_h_to_4h'


class FFNNGateProjectionAttr(Enum):
    GATE_PROJ = 'gate_proj'


class FFNNActivationFunctionAttr(Enum):
    ACT_FN = 'act_fn'
    ACT = 'act'


class FFNNDownProjectionAttr(Enum):
    DOWN_PROJ = 'down_proj'
    C_PROJ = 'c_proj'
    DENSE_4H_TO_H = 'dense_4h_to_h'


class FFNNDropoutAttr(Enum):
    DROPOUT = 'dropout'


class LayerInitialNormAttr(Enum):
    INPUT_LAYERNORM = 'input_layernorm'
    LN_1 = 'ln_1'


class LayerAttentionAttr(Enum):
    SELF_ATTN = 'self_attn'
    ATTN = 'attn'
    ATTENTION = 'attention'


class LayerAttentionDropoutAttr(Enum):
    POST_ATTENTION_DROPOUT = 'post_attention_dropout'


class LayerIntermediateNormAttr(Enum):
    INPUT_LAYERNORM = 'post_attention_layernorm'
    LN_2 = 'ln_2'


class LayerFeedForwardAttr(Enum):
    MLP = 'mlp'


class LayerFeedForwardDropoutAttr(Enum):
    POST_MLP_DROPUT = 'post_mlp_dropout'


class TransformerLayersAttr(Enum):
    LAYERS = 'layers'
    H = 'h'


class TransformerNormAttr(Enum):
    NORM = 'norm'
    LN_F = 'ln_f'
    FINAL_LAYER_NORM = 'final_layer_norm'


class LMTransformerAttr(Enum):
    MODEL = 'model'
    TRANSFORMER = 'transformer'
    GPT_NEOX = 'gpt_neox'


class LMHeadAttr(Enum):
    LM_HEAD = 'lm_head'
    EMBED_OUT = 'embed_out'


AttrEnumTypes: Type = Union[
    AttnQKVProjectionAttr, AttnOutProjectionAttr, AttnDropoutAttr,
    FFNNGateProjectionAttr, FFNNUpProjectionAttr, FFNNDownProjectionAttr, FFNNActivationFunctionAttr, FFNNDropoutAttr,
    LayerInitialNormAttr, LayerAttentionAttr, LayerAttentionDropoutAttr, LayerIntermediateNormAttr, LayerFeedForwardAttr, LayerFeedForwardDropoutAttr,
    TransformerEmbeddingAttr, TransformerPositionEmbeddingAttr, TransformerLayersAttr, TransformerNormAttr,
    LMTransformerAttr, LMHeadAttr
]

MultiAttrEnumTypes: Type = Union[AttnQKVProjectionsAttr]
