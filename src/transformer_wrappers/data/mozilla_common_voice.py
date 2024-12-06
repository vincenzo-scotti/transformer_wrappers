import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import Dataset

from typing import Optional, Dict, List, Union, Iterable, Tuple


class MozillaCommonVoice(Dataset):
    DURATIONS_FILE: str = 'clip_durations.tsv'
    MAX_CHAR: int = 200  # 200
    MAX_DURATION: int = 12000  # 12000
    MIN_DURATION: int = 1000  # 250
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
            language: Optional[Union[str, Iterable[str]]] = None,
            subsample: Optional[Dict[str, float]] = None,
            random_seed: Optional[int] = None
    ):
        #
        subsample = subsample if subsample is not None else dict()
        #
        self.paths: Iterable[str] = [path] if isinstance(path, str) else path
        self.split: str = split
        if language is None:
            language = set(
                lang_id for path in self.paths for lang_id in os.listdir(path)
                if os.path.isdir(os.path.join(path, lang_id))
            )
        self.languages: Tuple[str] = (language,) if isinstance(language, str) else tuple(set(language))
        self.subsample: Dict[str, float] = subsample if subsample is not None else None
        self.random_seed: Optional[int] = random_seed
        #
        data: List[pd.DataFrame] = list()
        for path in self.paths:
            for language in self.languages:
                if os.path.exists(os.path.join(path, language, self._split_mapping[self.split])):
                    df = pd.read_csv(os.path.join(path, language, self._split_mapping[self.split]), sep='\t')
                    df_durations = pd.read_csv(os.path.join(path, language, self.DURATIONS_FILE), sep='\t')
                    df = df.join(df_durations.rename(columns={'clip': 'path'}).set_index('path'), on='path')
                    df = df[(df['duration[ms]'] > self.MIN_DURATION) & (df['duration[ms]'] <= self.MAX_DURATION)]
                    df = df[df.apply(lambda r: len(r['sentence']) <= self.MAX_CHAR, axis=1)]
                    fraction = self.subsample.get(language, 1.0)
                    if 0.0 < fraction < 1.0:
                        gss = GroupShuffleSplit(n_splits=1, train_size=fraction, random_state=self.random_seed)
                        groups = df['client_id']
                        idxs, _ = next(gss.split(df, groups=groups))
                        df = df.iloc[idxs]
                    df['language'] = language
                    df['base_path'] = os.path.join(path, language)
                    data.append(df)
        self.data: pd.DataFrame = pd.concat(data, ignore_index=True) if len(data) > 1 else data[0]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data.iloc[index].to_dict()
