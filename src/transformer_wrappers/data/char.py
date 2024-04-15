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
        # tmp = '[INST] I\'m learning about set notation. What is the difference between a co-domain and an image? [/INST]In mathematics, the terms "co-domain" and "image" are related to the concept of a function, which is a rule that assigns an output value to each input value.\n\nThe co-domain of a function is the set of all possible output values that the function can produce. It is usually defined as a part of the function\'s definition, and it can be larger than the set of actual output values (i.e., the image) produced by the function. In other words, the co-domain represents the set of all possible values that the function could output, whether or not those values are actually output for any specific input value.\n\nThe image of a function, on the other hand, is the set of all actual output values that the function produces when given a particular set of input values. In other words, the image is the subset of the co-domain that actually gets used by the function for some input values. It is the set of all possible values that the function actually outputs for some input values.\n\nTo summarize, the co-domain is the set of all possible output values that a function can produce, while the image is the set of actual output values that the function produces for some input values. [INST] Can you explain the basics of set notation to me? [/INST]Certainly! In mathematics, a set is a collection of distinct objects, called elements, that share some common property. Set notation is a way of representing sets using symbols and mathematical notation.\n\nHere are some basics of set notation:\n\n1. Set notation is often denoted using braces: { }, which contain the elements of the set separated by commas. For example, the set of natural numbers less than 5 can be written as {0, 1, 2, 3, 4}.\n\n2. Sometimes, we use ellipses to represent a pattern or sequence of elements. For example, the set of even natural numbers can be written as {2, 4, 6, ...}.\n\n3. If an element is in a set, we use the symbol "∈" to show that it belongs to that set. For example, 3 ∈ {1, 2, 3, 4}.\n\n4. If an element is not in a set, we use the symbol "<unk>" to show that it does not belong to that set. For example, 5 <unk> {1, 2, 3, 4}.\n\n5. We can use set-builder notation to define a set by specifying a property that its elements share. For example, the set of even numbers can be written as {x : x is an integer and x is even}.\n\n6. We use mathematical symbols to denote the relationship between sets. For example, the symbol "<unk>" (subset) is used to show that one set is a subset of another. For example, {1, 2, 3} <unk> {1, 2, 3, 4, 5}.\n\nI hope this helps!</s>'
        tmp = 'I\'m learning about set notation. What is the difference between a co-domain and an image?'
        self.data = [x for x in self.data if tmp not in x]
        # End of TODO

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    @torch.no_grad()
    def collate(
            self, samples: List[str]
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        old_input_ids = self.tokenizer(samples)['input_ids']
        input_ids, valid_mask = self.char_tokenizer(
            self.tokenizer.batch_decode(old_input_ids),
            return_tensors='pt',
            padding=True,
            padding_side=self.tokenizer.padding_side
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
