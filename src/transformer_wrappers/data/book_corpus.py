import re
import numpy as np
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict


class BookCorpus(Dataset):
    # TODO make better version of this code
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

        escaped_token_regex = re.compile(r'^<0x\w\w>$')

        avg_token_len: int = int(
            np.ceil(np.mean([
                len(token) if not escaped_token_regex.match(token) else 1 for token in
                self.tokenizer.get_vocab().keys()
            ]))
        )
        self.max_seq_len_char: int = self.max_seq_len * avg_token_len

        tmp_data = load_dataset('bookcorpus/bookcorpus', split=self.split)

        self.data: str = '\n'.join(tmp_data['text'])

    def __len__(self) -> int:
        # Number of sequences within the data set
        return int(np.ceil(len(self.data) / self.max_seq_len_char))

    def __getitem__(self, index: int) -> Dict:
        return {'text': self.data[index * self.max_seq_len_char:(index + 1) * self.max_seq_len_char]}
