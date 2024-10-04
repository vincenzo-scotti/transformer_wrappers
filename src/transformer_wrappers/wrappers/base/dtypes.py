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
    'LayerIntermediateNormAttr',
    'LayerFeedForwardAttr',
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


class TransformerPositionEmbeddingAttr(Enum):
    WPE = 'wpe'


class AttnQKVProjectionAttr(Enum):
    C_ATTN = 'c_attn'


class AttnQKVProjectionsAttr(Enum):
    QKV_ATTN = ('q_proj', 'k_proj', 'v_proj')


class AttnOutProjectionAttr(Enum):
    C_PROJ = 'c_proj'
    O_PROJ = 'o_proj'


class AttnDropoutAttr(Enum):
    DROPOUT = 'dropout'


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
    AttnQKVProjectionAttr, AttnOutProjectionAttr, AttnDropoutAttr,
    FFNNGateProjectionAttr, FFNNUpProjectionAttr, FFNNDownProjectionAttr, FFNNActivationFunctionAttr, FFNNDropoutAttr,
    LayerInitialNormAttr, LayerAttentionAttr, LayerIntermediateNormAttr, LayerFeedForwardAttr,
    TransformerEmbeddingAttr, TransformerPositionEmbeddingAttr, TransformerLayersAttr, TransformerNormAttr,
    LMTransformerAttr, LMHeadAttr
]

MultiAttrEnumTypes: Type = Union[AttnQKVProjectionsAttr]
