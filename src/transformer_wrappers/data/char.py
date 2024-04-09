import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import List, Tuple, Callable, Optional, Dict


class TokeNNDataset(Dataset):
    # TODO make this code more general
    def __init__(
            self,
            split: str,
            embeddings: nn.Embedding,
            tokenizer: PreTrainedTokenizer,
            char_tokenizer: Callable,
            max_seq_len: Optional[int] = None,
            device: Optional[torch.device] = None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = [
            tokenizer.decode(tokenizer.apply_chat_template([
                {
                    'role': 'user' if message.split(':', 1)[0].strip() == 'Human' else 'assistant',
                    'content': message.split(':', 1)[1].strip()
                }
                for message in dialogue.split('###')
                if len(message) > 0
            ])[:self.max_seq_len if self.max_seq_len is not None else -1])
            for dialogue in load_dataset('timdettmers/openassistant-guanaco', split=self.split)['text']
        ]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def collate(
            self, samples: List[str]
    ) -> Tuple[Dict[torch.tensor, torch.tensor], torch.tensor, torch.tensor, torch.tensor]:
        input_encodings = self.char_tokenizer(samples, return_tensors='pt', device=self.device)
        tgt_input_encodings = self.tokenizer(samples, return_tensors='pt', padding=True).to(self.device)
        tgt_embeddings = self.embeddings(tgt_input_encodings.input_ids)
        tgt_attention_mask = tgt_input_encodings.attention_mask
        tgt_out_gate = self.char_tokenizer.get_out_gate(
            [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in tgt_input_encodings.input_ids],
            return_tensors='pt',
            device=self.device
        )

        mini_batch = input_encodings, tgt_embeddings, tgt_out_gate, tgt_attention_mask

        return mini_batch
