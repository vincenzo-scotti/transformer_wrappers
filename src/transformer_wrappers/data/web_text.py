from itertools import islice

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from typing import Optional, Dict


class OpenWebText(Dataset):
    # TODO make better version of this code
    def __init__(
            self,
            split: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: Optional[int] = None,
            n_docs: Optional[int] = None,
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len
        self.n_docs: Optional[int] = n_docs

        data = load_dataset('Skylion007/openwebtext', streaming=True, split=self.split, trust_remote_code=True)
        if self.split == 'train':
            data = data.shuffle()

        if self.n_docs is not None:
            self.data = [
                {
                    'text': self.tokenizer.decode(self.tokenizer(
                        sample['text'] + self.tokenizer.eos_token, truncation=True, max_length=self.max_seq_len
                    )['input_ids'])
                }
                for sample in islice(data, self.n_docs)
            ]
        else:
            self.data = data

    def __len__(self) -> int:
        # Number of sequences within the data set
        return self.n_docs if self.n_docs is not None else self.data.info.splits[self.split].num_examples

    def _get_item_steaming(self, index: int) -> Dict:
        return {'text': self.tokenizer.decode(
            self.tokenizer(
                (next(iter(self.data.filter(lambda sample, idx: idx == index, with_indices=True)))['text'] +
                 # self.tokenizer.eos_token,
                 self.tokenizer.eos_token)[:self.max_seq_len * 10],
                truncation=True,
                max_length=self.max_seq_len
            )['input_ids']
        )}

    def __getitem__(self, index: int) -> Dict:
        if self.n_docs is not None:
            return self.data[index]
        else:
            return self._get_item_steaming(index)
