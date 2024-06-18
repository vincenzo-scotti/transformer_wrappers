from itertools import batched, chain

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict, List, Union


class BookCorpus(Dataset):
    # TODO make this code more general
    def __init__(
            self,
            split: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: Optional[int] = None,
            eval_size: Union[int, float] = 1000,
            seed: Optional[int] = None
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len
        self.eval_size: Union[int, float] = eval_size
        self.seed: Optional[int] = seed

        tmp_data = load_dataset('bookcorpus/bookcorpus', split='train')
        train_idxs, test_idxs = train_test_split(range(len(tmp_data)), test_size=self.eval_size, random_state=self.seed)
        train_idxs, validation_idxs = train_test_split(train_idxs, test_size=self.eval_size, random_state=self.seed)

        if self.split == 'train':
            idxs = train_idxs
        elif self.split == 'validation':
            idxs = validation_idxs
        elif self.split == 'test':
            idxs = test_idxs
        else:
            raise ValueError(f'Unknown split: `{self.split}`')

        self.data: List[Dict[str, str]] = [
            {'text': tokenizer.decode(seq)}
            for seq in batched(
                chain.from_iterable(
                    tokenizer([sample + tokenizer.eos_token for sample in tmp_data[idxs]['text']])['input_ids']
                ),
                self.max_seq_len
            )
        ]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]
