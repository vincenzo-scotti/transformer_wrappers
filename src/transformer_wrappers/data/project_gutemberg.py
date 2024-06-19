from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict


class PG19(Dataset):
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

        self.data = load_dataset('deepmind/pg19', streaming=True, split=self.split)

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        raise NotImplementedError('Need to implement index handling within documents')
