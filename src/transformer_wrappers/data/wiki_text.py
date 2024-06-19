import re
import numpy as np
from itertools import batched, chain

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict, List


class _WikiText(Dataset):
    # TODO make this code more general

    _name: Optional[str] = None

    def __init__(
            self,
            split: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: Optional[int] = None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len

        tmp_data = load_dataset('wikitext', self._name, split=self.split)

        sep_symbol = self.tokenizer.eos_token
        if self.tokenizer.bos_token is not None:
            sep_symbol = sep_symbol + self.tokenizer.bos_token

        self.data: List[Dict[str, str]] = [
            {'text': tokenizer.decode(seq)}
            for seq in batched(
                chain.from_iterable(
                    self.tokenizer([
                        sample + self.tokenizer.eos_token
                        for sample in re.sub(
                            r'\n = ([^=])', f'\n{sep_symbol} = \\1', '\n'.join(tmp_data['text']).lstrip('\n')
                        ).split(sep_symbol)
                    ])['input_ids']
                ),
                self.max_seq_len
            )
        ]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]


class WikiText2(_WikiText):
    _name: str = 'wikitext-2-raw-v1'

class WikiText103(_WikiText):
    _name: str = 'wikitext-103-raw-v1'

    def __init__(
            self,
            split: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: Optional[int] = None
    ):
        if split == 'train':
            super(Dataset, self).__init__()

            self.split = split
            self.tokenizer: PreTrainedTokenizer = tokenizer
            self.max_seq_len: Optional[int] = max_seq_len

            escaped_token_regex = re.compile(r'^<0x\w\w>$')
            avg_token_len = int(
                np.ceil(np.mean([
                    len(token) if not escaped_token_regex.match(token) else 1 for token in
                    self.tokenizer.get_vocab().keys()
                ]))
            )

            sep_symbol = self.tokenizer.eos_token
            if self.tokenizer.bos_token is not None:
                sep_symbol = sep_symbol + self.tokenizer.bos_token

            tmp_data = load_dataset('wikitext', self._name, split=self.split)
            tmp_data = (
                    self.tokenizer.bos_token +
                    re.sub(r'\n = ([^=])', f'\n{sep_symbol} = \\1', '\n'.join(tmp_data['text']).lstrip('\n')) +
                    self.tokenizer.eos_token
            )

            self.data: List[Dict[str, str]] = [
                {'text': tmp_data[i:i + (avg_token_len * self.max_seq_len)]}
                for i in range(0, len(tmp_data), avg_token_len * self.max_seq_len)
            ]
        else:
            super().__init__(split, tokenizer, max_seq_len=max_seq_len)
