from .base import PreTrainedModelWrapper, TransformerWrapper, CausalLMWrapper
from .parallel import ParallelTransformerWrapper, ParallelCausalLMWrapper
from .resizable import ResizableTokenizer, ResizableTransformerWrapper, ResizableCausalLMWrapper
from .speech import SpeechTransformerWrapper, SpeechCausalLMWrapper

from typing import Dict, Type


transformer_mapping: Dict[str, Type[TransformerWrapper]] = {
    TransformerWrapper.__name__: TransformerWrapper,
    ParallelTransformerWrapper.__name__: ParallelTransformerWrapper,
    ResizableTransformerWrapper.__name__: ResizableTransformerWrapper,
    SpeechTransformerWrapper.__name__: SpeechTransformerWrapper
}

causal_lm_mapping: Dict[str, Type[CausalLMWrapper]] = {
    CausalLMWrapper.__name__: CausalLMWrapper,
    ParallelCausalLMWrapper.__name__: ParallelCausalLMWrapper,
    ResizableCausalLMWrapper.__name__: ResizableCausalLMWrapper,
    SpeechCausalLMWrapper.__name__: SpeechCausalLMWrapper
}
