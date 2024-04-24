import re

import torch

from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import logging

from typing import Type, Optional, Dict, Set, List, Pattern

from .base import ModuleWrapper, TransformerWrapper, CausalLMWrapper


__all__ = ['ResizableTokenizer', 'ResizableTransformerWrapper', 'ResizableLMHeadWrapper']


MAX_TOKEN_LEN: str = 'max_token_len'


logger = logging.get_logger(__name__)


class ResizableTokenizer(PreTrainedTokenizer):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_token_len: Optional[int] = None,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_token_len: Optional[int] = max_token_len

        self._escaped_token_regex: Pattern[str] = re.compile(r'^<0x\w\w>$')

        super().__init__(
            model_max_length=tokenizer.model_max_length,
            padding_side=tokenizer.padding_side,
            truncation_side=tokenizer.truncation_side,
            chat_template=tokenizer.chat_template,
            model_input_names=tokenizer.model_input_names,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            unk_token=tokenizer.unk_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token,
            cls_token=tokenizer.cls_token,
            mask_token=tokenizer.mask_token,
            additional_special_tokens=tokenizer.additional_special_tokens,
            clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
            split_special_tokens=tokenizer.split_special_tokens
        )

    @classmethod
    def from_pretrained(cls, *args, max_token_len: Optional[int] = None, **kwargs) -> 'ResizableTokenizer':
        return cls(AutoTokenizer.from_pretrained(*args, **kwargs), max_token_len=max_token_len)

    """
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        batch_text_or_text_pairs = self.tokenizer.batch_decode(self.tokenizer(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )['input_ids'])
        return super()._batch_encode_plus(batch_text_or_text_pairs, add_special_tokens=add_special_tokens, **kwargs)

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            **kwargs
    ) -> BatchEncoding:
        text = self.tokenizer.decode(self.tokenizer(text, add_special_tokens=add_special_tokens)['input_ids'])
        if text_pair is not None:
            text_pair = self.tokenizer.decode(tokenizer(text_pair, add_special_tokens=add_special_tokens)['input_ids'])
        return super()._encode_plus(
            text, text_pair=text_pair, add_special_tokens=add_special_tokens, **kwargs
        )
    """

    def build_inputs_with_special_tokens(self, *args, **kwargs):
        return self.tokenizer.build_inputs_with_special_tokens(*args, **kwargs)

    def get_vocab(self, max_token_len: Optional[int] = None) -> Dict[str, int]:
        max_token_len = max_token_len if max_token_len is not None else self.max_token_len

        if max_token_len is None:
            return self.tokenizer.get_vocab()
        else:
            special_tokens: Set[str] = set(self.special_tokens_map.values())  # TODO handle additional special tokens
            return {
                token: idx for token, idx in self.tokenizer.get_vocab().items()
                if token in special_tokens or self._escaped_token_regex.match(token) or len(token) <= max_token_len
            }

    def add_tokens(self, *args, **kwargs):
        return self.tokenizer.add_tokens(*args, **kwargs)

    def add_special_tokens(self, *args, **kwargs):
        return self.tokenizer.add_special_tokens(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.convert_tokens_to_ids(self.tokenize(*args, **kwargs))

    def push_to_hub(self, *args, **kwargs):
        return self.tokenizer.push_to_hub(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def get_added_vocab(self):
        return self.tokenizer.get_added_vocab()

    def num_special_tokens_to_add(self, *args, **kwargs):
        return self.tokenizer.num_special_tokens_to_add(*args, **kwargs)

    def prepare_for_tokenization(self, *args, **kwargs):
        return self.tokenizer.prepare_for_tokenization(*args, **kwargs)

    def _split_token(self, token: str, vocabulary: Set[str]) -> List[str]:
        if token in vocabulary:
            return [token]

        split = list(token)
        for i in range(1, len(token) - 1):
            tmp_split = self._split_token(token[:i], vocabulary) + self._split_token(token[i:], vocabulary)
            if len(tmp_split) < len(split):
                split = tmp_split

        return split

    def tokenize(self, *args, max_token_len: Optional[int] = None, **kwargs) -> List[str]:
        max_token_len = max_token_len if max_token_len is not None else self.max_token_len

        if max_token_len is None:
            return self.tokenizer.tokenize(*args, **kwargs)
        elif max_token_len < 0:
            raise ValueError()

        vocabulary: Set[str] = set(self.get_vocab())
        tokens = list()

        # TODO rewrite to be more efficient
        for token in self.tokenizer.tokenize(*args, **kwargs):
            if token in vocabulary:
                tokens.append(token)
            else:
                tokens.extend(self._split_token(token, vocabulary))

        return tokens


class ResizableTransformerWrapper(TransformerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.max_token_len: Optional[int] = self.config.task_specific_params[
            self.WRAPPER_CONFIGS_KEY
        ].get(MAX_TOKEN_LEN)
        self._tokenizer = ResizableTokenizer(self._tokenizer, max_token_len=self.max_token_len)

    @property
    def max_token_len(self):
        return self._max_token_len

    @max_token_len.setter
    def max_token_len(self, max_token_len: Optional[int] = None):
        self._max_token_len = max_token_len
        self._tokenizer.max_token_len = max_token_len


class ResizableLMHeadWrapper(ModuleWrapper):
    def _apply_logits_mask(self, logits: torch.tensor) -> torch.tensor:
        mask = torch.full_like(logits, torch.finfo(logits.dtype).min)
        mask[..., list(self.super_wrapper.tokenizer.get_vocab().values())] = 0

        return logits + mask

    def _wrapped_forward(self, **kwargs):
        output = super()._wrapped_forward(**kwargs)
        output[self.module_output] = self._apply_logits_mask(output[self.module_output])

        return output


class ResizableCausalLMWrapper(CausalLMWrapper):
    _lm_head_dtype: Type[ModuleWrapper] = ResizableLMHeadWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.max_token_len: Optional[int] = self.config.task_specific_params[
            self.WRAPPER_CONFIGS_KEY
        ].get(MAX_TOKEN_LEN)
        self._tokenizer = ResizableTokenizer(self._tokenizer, max_token_len=self.max_token_len)

    @property
    def max_token_len(self):
        return self._max_token_len

    @max_token_len.setter
    def max_token_len(self, max_token_len: Optional[int] = None):
        self._max_token_len = max_token_len
        self._tokenizer.max_token_len = max_token_len


"""
Old code

def split_token(token: str, vocabulary: Set[str]) -> List[str]:
    if token in vocabulary:
        return [token]

    split = list(token)
    for i in range(1, len(token) - 1):
        tmp_split = split_token(token[:i], vocabulary) + split_token(token[i:], vocabulary)
        if len(tmp_split) < len(split):
            split = tmp_split

    return split


def encode(text: str, tokenizer: PreTrainedTokenizer, max_token_len: Optional[int] = None) -> List[int]:
    if max_token_len is None:
        return tokenizer(text)['input_ids']
    elif max_token_len < 0:
        raise ValueError()

    escaped_token_regex: Pattern[str] = re.compile(r'^<0x\w\w>$')
    special_tokens: Set[str] = set(tokenizer.special_tokens_map.values())  # TODO handle additional special tokens
    vocabulary: Set[str] = set(
        token for token in tokenizer.vocab
        if token in special_tokens or escaped_token_regex.match(token) or len(token) <= max_token_len
    )

    input_ids = list()

    for idx in tokenizer(text)['input_ids']:
        token: str = tokenizer.convert_ids_to_tokens(idx)

        if token in vocabulary:
            input_ids.append(idx)
        else:
            input_ids.extend(tokenizer.convert_tokens_to_ids(split_token(token, vocabulary)))

    return input_ids




tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
input_ids = encode('Hello, world!', tokenizer)
print(tokenizer.convert_ids_to_tokens(input_ids))
input_ids = encode('Hello, world!', tokenizer, max_token_len=3)
print(tokenizer.convert_ids_to_tokens(input_ids))

"""
