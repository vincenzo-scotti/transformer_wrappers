import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, BatchEncoding

from datasets import load_dataset

from typing import List, Tuple, Optional


class OpenAssistantGuanaco(Dataset):

    # TODO make this code more general
    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, max_seq_len: Optional[int] = None):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len

        sep_regex = re.compile(r'### (Human|Assistant)')

        self.data = [
            tokenizer.decode(tokenizer.apply_chat_template([
                {
                    'role': 'user' if message.split(':', 1)[0].strip() == 'Human' else 'assistant',
                    'content': message.split(':', 1)[1].strip()
                }
                for message in sep_regex.sub('<sep/> \\1', dialogue).split('<sep/>')
                if len(message) > 0
            ]), skip_special_tokens=True).strip() + tokenizer.eos_token
            for idx, dialogue in enumerate(load_dataset(
                'timdettmers/openassistant-guanaco',
                split=self.split if split != 'validation' else 'train'
            )['text'])
            if (
                self.split == 'test' or
                (self.split == 'validation' and idx % 10 == 0) or
                (self.split == 'train' and idx % 10 != 0)
            )
        ]
        # End of TODO

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    @torch.no_grad()
    def collate(
            self, samples: List[str]
    ) -> Tuple[BatchEncoding, torch.tensor]:
        input_encodings = self.tokenizer(
            samples,
            return_tensors='pt',
            padding=True,
            truncation=self.max_seq_len is not None,
            max_length=self.max_seq_len
        )
        output_ids = input_encodings.input_ids.clone()
        output_ids[~input_encodings.attention_mask.bool()] = -100

        return input_encodings, output_ids
