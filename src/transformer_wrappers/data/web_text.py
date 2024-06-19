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
            max_seq_len: Optional[int] = None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len

        self.data = load_dataset('Skylion007/openwebtext', streaming=True, split=self.split)

    def __len__(self) -> int:
        # Number of sequences within the data set
        return self.data.info.splits[self.split].num_examples

    def __getitem__(self, index: int) -> Dict:
        return {'text': self.tokenizer.decode(
            self.tokenizer(
                next(iter(self.data.filter(lambda sample, idx: idx == index, with_indices=True)))['text'] +
                self.tokenizer.eos_token,
                truncation=True,
                max_length=self.max_seq_len
            )['input_ids']
        )}
