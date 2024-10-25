import os

import pandas as pd

from torch.utils.data import Dataset

from typing import Optional, Dict, List, Union, Iterable, Tuple


class MozillaCommonVoice(Dataset):
    _split_mapping: Dict[str, str] = {
        'train': 'train.tsv',
        'validation': 'dev.tsv',
        'test': 'test.tsv'
    }

    # TODO make this code more general
    def __init__(
            self,
            path: Union[str, Iterable[str]],
            split: str,
            language: Optional[Union[str, Iterable[str]]] = None
    ):
        #
        self.paths: Iterable[str] = [path] if isinstance(path, str) else path
        self.split: str = split
        if language is None:
            language = set(
                lang_id for path in self.paths for lang_id in os.listdir(path)
                if os.path.isdir(os.path.join(path, lang_id))
            )
        self.languages: Tuple[str] = (language,) if isinstance(language, str) else tuple(set(language))
        #
        data: List[pd.DataFrame] = list()
        for path in self.paths:
            for language in self.languages:
                if os.path.exists(os.path.join(path, language, self._split_mapping[self.split])):
                    df = pd.read_csv(os.path.join(path, language, self._split_mapping[self.split]), sep='\t')
                    df['language'] = language
                    df['base_path'] = os.path.join(path, language)
                    data.append(df)
        self.data: pd.DataFrame = pd.concat(data) if len(data) > 1 else data[0]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data.iloc[index].to_dict()
