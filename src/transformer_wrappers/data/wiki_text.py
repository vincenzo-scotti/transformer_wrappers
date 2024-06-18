import re
from itertools import batched

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict, List


class WikiText2(Dataset):
    # TODO make this code more general
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

        tmp_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=self.split)

        sep_symbol = self.tokenizer.eos_token
        if self.tokenizer.bos_token is not None:
            sep_symbol = sep_symbol + self.tokenizer.bos_token

        self.data: List[Dict[str, str]] = [
            {'text': tokenizer.decode(seq)}
            for seq in batched(
                self.tokenizer(
                    re.sub(r'\n = ([^=])', f'\n{sep_symbol} = \\1', '\n'.join(tmp_data['text']).lstrip('\n')) +
                    tokenizer.eos_token
                )['input_ids'],
                self.max_seq_len
            )
        ]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]
