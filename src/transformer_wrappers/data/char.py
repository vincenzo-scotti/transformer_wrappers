import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer
from transformer_wrappers.wrappers.char import CharTokenizer

from datasets import load_dataset

from typing import List, Tuple, Optional


class TokeNNDataset(Dataset):

    # TODO make this code more general
    def __init__(
            self,
            split: str,
            embeddings: nn.Embedding,
            tokenizer: PreTrainedTokenizer,
            char_tokenizer: CharTokenizer,
            max_seq_len: Optional[int] = None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.embeddings: nn.Embedding = embeddings
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.char_tokenizer: CharTokenizer = char_tokenizer
        self.max_seq_len: Optional[int] = max_seq_len

        # TODO make this code more modular
        sep_regex = re.compile(r'### (Human|Assistant)')

        self.data = [
            tokenizer.decode(tokenizer.apply_chat_template([
                {
                    'role': 'user' if message.split(':', 1)[0].strip() == 'Human' else 'assistant',
                    'content': message.split(':', 1)[1].strip()
                }
                for message in sep_regex.sub('<sep/> \\1', dialogue).split('<sep/>')
                if len(message) > 0
            ]), skip_special_tokens=True).strip()
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

    def collate(
            self, samples: List[str]
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        old_input_ids = self.tokenizer(samples)['input_ids']
        input_ids, valid_mask = self.char_tokenizer(
            self.tokenizer.batch_decode(old_input_ids), return_tensors='pt'
        ).values()
        tgt_input_encodings = self.tokenizer(samples, return_tensors='pt', padding=True)
        tgt_embeddings = self.embeddings(tgt_input_encodings.input_ids)
        tgt_attention_mask = tgt_input_encodings.attention_mask
        tgt_out_gate = self.char_tokenizer.get_out_gate(
            [
                [
                    chr(int(CharTokenizer.escaped_tokens_regex.match(s)[1], 16))
                    if CharTokenizer.escaped_tokens_regex.match(s)
                    else s
                    for s in self.tokenizer.convert_ids_to_tokens(input_ids)
                ]
                for input_ids in old_input_ids
            ],
            return_tensors='pt'
        )

        mini_batch = input_ids, valid_mask, tgt_embeddings, tgt_out_gate, tgt_attention_mask

        return mini_batch
