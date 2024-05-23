import re

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from datasets import load_dataset

from typing import Optional, Dict, List, Union


class OpenAssistantGuanaco(Dataset):

    # TODO make this code more general
    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, max_seq_len: Optional[int] = None):
        super(Dataset, self).__init__()

        self.split = split
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len: Optional[int] = max_seq_len

        sep_regex = re.compile(r'### (Human|Assistant)')

        self.data: List[Dict[str, Union[str, List[Dict[str, str]]]]] = [
            {
                'messages': [
                    {
                        'role': 'user' if message.split(':', 1)[0].strip() == 'Human' else 'assistant',
                        'content': message.split(':', 1)[1].strip()
                    }
                    for message in sep_regex.sub('<sep/> \\1', dialogue).split('<sep/>') if len(message) > 0
                ]
            }
            for idx, dialogue in enumerate(load_dataset(
                'timdettmers/openassistant-guanaco', split=self.split if split != 'validation' else 'train'
            )['text'])
            if (
                self.split == 'test' or
                (self.split == 'validation' and idx % 10 == 0) or
                (self.split == 'train' and idx % 10 != 0)
            )
        ]
        for sample in self.data:
            sample['text'] = tokenizer.decode(
                tokenizer.apply_chat_template(sample['messages']), skip_special_tokens=True
            ).strip() + tokenizer.eos_token

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]
