import os
from itertools import product
import random
from jinja2 import Environment, Template

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from typing import Optional, Dict, List, Union, Iterable, Tuple, Literal


__all__ = ['MozillaCommonVoice', 'ProcessedMozillaCommonVoice']


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
        self.data: pd.DataFrame = pd.concat(data) if len(data) > 1 else data

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data.iloc[index].to_dict()


class ProcessedMozillaCommonVoice(MozillaCommonVoice):
    _tasks: Tuple[str, str, str] = ('speak', 'rec', 'asr', 'tts')
    _jinja_env: Environment = Environment()

    LANGUAGES: Tuple[str, ...] = ('en', 'it')
    _languages_dict: Dict[str, Dict[str, str]] = {
        'en': {'en': 'English', 'it': 'Italian'}, 'it': {'en': 'inglese', 'it': 'italiano'},
    }

    SPEAK_TEMPLATE: str = '{% for sample in samples %}{% if loop.index0 > 0 and tokenizer.bos_token is not none %}{{ tokenizer.bos_token }}{% else if loop.index0 == 0 and tokenizer.bos_token is none %}{{ tokenizer.eos_token }}{% endif %}{{ audio_token }}{{ tokeniser.eos_token }}{% endfor %}'
    REC_INSTRUCTIONS: Dict[str, List[str]] = {
        'en': [
                  f'{command} the {target} in the{" following" if following else ""} {source}{eos}'
                  for command, target, following, source, eos in [
                *product(
                    ['Identify', 'Recognize'],
                    ['language', 'language spoken', 'language being spoken'],
                    [True, False],
                    ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                     'recording{% if samples | length > 1 %}s{% endif %}'],
                    ['.']
                ),
                *product(
                    ['What is'],
                    ['language', 'language spoken', 'language being spoken'],
                    [True, False],
                    ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                     'recording{% if samples | length > 1 %}s{% endif %}'],
                    ['?']
                )
            ]
              ] + [
                  f'{command} the {target}{eos}'
                  for command, target, eos in [
                *product(
                    ['Identify', 'Recognize'],
                    ['language', 'language spoken', 'language being spoken'],
                    ['']
                ),
                *product(
                    ['What is'],
                    ['language', 'language spoken', 'language being spoken'],
                    ['?']
                )
            ]
              ] + [
                  f'The following '
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}are pairs{{% else %}}is a pair{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}are gropus{{% else %}}is a pair{{% endif %}}'
                  f'{{% endif %}} '
                  f'of {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'and {{% if samples | length > 1 %}}their{{% else %}}its{{% endif %}} corresponding langauage{{% if samples | length > 1 %}}s{{% endif %}} being spoken.'
                  for source in ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                                 'recording{% if samples | length > 1 %}s{% endif %}']
              ],
        'it': [
                  f'{command} la {target} nell{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{eos}'
                  for command, target, following, source, eos in [
                *product(
                    ['Identifica', 'Riconosci'],
                    ['lingua', 'lingua parlata'],
                    [True, False],
                    ['clip audio', 'registrazion{% if samples | length > 1 %}i{% else %}e{% endif %}'],
                    ['.']
                ),
                *product(
                    ['Qual è'],
                    ['lingua', 'lingua parlata'],
                    [True, False],
                    ['clip audio', 'recording{% if samples | length > 1 %}s{% endif %}'],
                    ['?']
                )
            ]
              ] + [
                  f'{command} la {target} ne{{% if samples | length > 1 %}}i{{% else %}}l{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{eos}'
                  for command, target, following, source, eos in [
                *product(
                    ['Identifica', 'Riconosci'],
                    ['lingua', 'lingua parlata'],
                    [True, False],
                    ['audio'],
                    ['.']
                ),
                *product(
                    ['Qual è'],
                    ['lingua', 'lingua parlata'],
                    [True, False],
                    ['audio'],
                    ['?']
                )
            ]
              ] + [
                  f'{command} la {target}{eos}'
                  for command, target, eos in [
                *product(
                    ['Identifica', 'Riconosci'],
                    ['lingua', 'lingua parlata'],
                    ['']
                ),
                *product(
                    ['Qual è'],
                    ['lingua', 'lingua parlata'],
                    ['?']
                )
            ]
              ] + [
                  f'L{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}} seguent{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}}'
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}sono coppie{{% else %}}è una coppia{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}sono gruppi{{% else %}}è una coppia{{% endif %}}'
                  f'{{% endif %}} '
                  f'di {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'con {{% if samples | length > 1 %}}le loro{{% else %}}la sua{{% endif %}} corrispondent{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}} lingu{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}} corrisppondent{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}}.'
                  for source in
                  ['audio', 'clip audio', 'registrazion{% if samples | length > 1 %}i{% else %}e{% endif %}']
              ]
    }
    ASR_INSTRUCTIONS: Dict[str, List[str]] = {
        'en': [
                  f'{command} the{" following" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Transcribe'],
                [True, False],
                ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                 'recording{% if samples | length > 1 %}s{% endif %}'],
                ['.']
            )
              ] + ['Transcribe'] + [
                  f'The following '
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}are pairs{{% else %}}is a pair{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}are gropus{{% else %}}is a pair{{% endif %}}'
                  f'{{% endif %}} '
                  f'of {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'and {{% if samples | length > 1 %}}their{{% else %}}its{{% endif %}} transcription{{% if samples | length > 1 %}}s{{% endif %}}.'
                  for source in ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                                 'recording{% if samples | length > 1 %}s{% endif %}']
              ],
        'it': [
                  f'{command} l{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Trascrivi'],
                [True, False],
                ['clip audio', 'registrazion{% if samples | length > 1 %}i{% else %}e{% endif %}'],
                ['.']
            )
              ] + [
                  f'{command} {{% if samples | length > 1 %}}i{{% else %}}il{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Trascrivi'],
                [True, False],
                ['audio'],
                ['.']
            )
              ] + ['Trascrivi'] + [
                  f'L{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}} seguent{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}}'
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}sono coppie{{% else %}}è una coppia{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}sono gruppi{{% else %}}è una coppia{{% endif %}}'
                  f'{{% endif %}} '
                  f'di {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'con {{% if samples | length > 1 %}}le loro{{% else %}}la sua{{% endif %}} trascrizion{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}}.'
                  for source in
                  ['audio', 'clip audio', 'registrazion{% if samples | length > 1 %}i{% else %}e{% endif %}']
              ]
    }
    TTS_INSTRUCTIONS: Dict[str, List[str]] = {
        'en': [
                  f'{command} the{" following" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Utter', 'Read'],
                [True, False],
                ['sentence{% if samples | length > 1 %}s{% endif %}',
                 'piece{% if samples | length > 1 %}s{% endif %} of text'],
                ['.']
            )
              ] + ['Utter', 'Read'] + [
                  f'The following '
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}are pairs{{% else %}}is a pair{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}are gropus{{% else %}}is a pair{{% endif %}}'
                  f'{{% endif %}} '
                  f'of {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'and {{% if samples | length > 1 %}}their{{% else %}}its{{% endif %}} corresponding {target}.'
                  for source, target in product(
                ['sentence{% if samples | length > 1 %}s{% endif %}',
                 'piece{% if samples | length > 1 %}s{% endif %} of text'],
                ['audio clip{% if samples | length > 1 %}s{% endif %}', 'audio',
                 'recording{% if samples | length > 1 %}s{% endif %}']
            )
              ],
        'it': [
                  f'{command} l{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Trascrivi'],
                [True, False],
                ['fras{% if samples | length > 1 %}i{% else %}e{% endif %}'],
                ['.']
            )
              ] + [
                  f'{command} {{% if samples | length > 1 %}}i{{% else %}}il{{% endif %}}{" seguent{% if samples | length > 1 %}i{% else %}e{% endif %}" if following else ""} {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{ {{ languages | getitem(sample.language) }} }}{{% endif %}}{eos}'
                  for command, following, source, eos in product(
                ['Pronuncia', 'Leggi'],
                [True, False],
                ['test{% if samples | length > 1 %}i{% else %}o{% endif %}'],
                ['.']
            )
              ] + ['Pronuncia', 'Leggi'] + [
                  f'L{{% if samples | length > 1 %}}e{{% else %}}a{{% endif %}} seguent{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}}'
                  f'{{% if approach == "paired" %}}'
                  f'{{% if samples | length > 1 %}}sono coppie{{% else %}}è una coppia{{% endif %}}'
                  f'{{% else %}}{{% if samples | length > 1 %}}sono gruppi{{% else %}}è una coppia{{% endif %}}'
                  f'{{% endif %}} '
                  f'di {source}{{% if info and samples is all(sample => sample.language == (samples | first).language) %}} in {{{{ languages | getitem(sample.language) }}}}{{% endif %}} '
                  f'e {{% if samples | length > 1 %}}le loro{{% else %}}la sua{{% endif %}} corrispondent{{% if samples | length > 1 %}}i{{% else %}}e{{% endif %}} {target}.'
                  for source, target in product(
                ['test{% if samples | length > 1 %}i{% else %}o{% endif %}',
                 'fras{% if samples | length > 1 %}i{% else %}e{% endif %}'],
                ['clip audio', 'registrazion{% if samples | length > 1 %}i{% else %}e{% endif %}']
            )
              ]
    }

    SAMPLE_START: List[str] = ['\n\n', '\n---\n']
    SAMPLES_SEP: List[str] = ['\n\n', '\n---\n']
    SAMPLE_IO_SEP: List[str] = [
        '\n\n',
        '{% if approach == "paired" %} {% else %}\n{% endif %}->{% if approach == "paired" %} {% else %}\n{% endif %}',
        '\n->\n',
        '{% if approach == "paired" %} {% else %}\n{% endif %}=={% if approach == "paired" %} {% else %}\n{% endif %}',
        '\n==\n',
        '{% if approach == "paired" %} {% else %}\n{% endif %}=>{% if approach == "paired" %} {% else %}\n{% endif %}',
        '\n=>\n'
    ]

    LANGUAGE_START: Dict[str, List[str]] = {
        'en': [
            'Language{% if approach == "split" and samples | length > 1 %} file {{ loop.index }}{% endif %}: ',
            'Detected langauge{% if approach == "split" and samples | length > 1 %} file {{ loop.index }}{% endif %}: '
        ],
        'it': [
            'Lingua{% if approach == "split" and samples | length > 1 %} file {{ loop.index }}{% endif %}: ',
            'Lingua identificata{% if approach == "split" and samples | length > 1 %} file {{ loop.index }}{% endif %}: '
        ]
    }
    LANGUAGE_FORMAT: List[str] = [
        '{{ sample.language }}', '{{ sample.language | upper }}', '{{ languages | getitem(sample.language) }}'
    ]
    AUDIO_PREFIX: Dict[str, List[str]] = {
        'en': ['Audio', 'Audio clip', 'Clip', 'Recording'],
        'it': ['Audio', 'Clip audio', 'Clip', 'Registrazione']
    }
    AUDIO_START: Dict[str, List[str]] = {
        lang_id: [
            f'{audio_prefix}{{% if (approach == "split" or numbered) and samples | length > 1 %}} {{{{loop.index}}}}{{% endif %}}: '
            for audio_prefix in AUDIO_PREFIX[lang_id]
        ]
        for lang_id in LANGUAGES
    }
    AUDIO_FORMAT: Dict[str, List[str]] = {
        'en': [
            '{0}{1}{{% if numbered and samples | length > 1 %}}_{{{{ loop.index }}}}{{% endif %}}{2}{{{{ audio_token }}}}{2}{0}'.format(
                boundary, data, "\n" if len(boundary) > 1 else "")
            for boundary, data in product(
                ['', '"', '\'', '`' '"""', '\'\'\'', '```'], ['audio', 'clip', 'recording']
            )
        ],
        'it': [
            '{0}{1}{{% if numbered and samples | length > 1 %}}_{{{{ loop.index }}}}{{% endif %}}{2}{{{{ audio_token }}}}{2}{0}'.format(
                boundary, data, "\n" if len(boundary) > 1 else "")
            for boundary, data in product(
                ['', '"', '\'', '`' '"""', '\'\'\'', '```'], ['audio', 'clip', 'registrazione']
            )
        ],
    }
    LANGUAGE_INFO_PREFIX: Dict[str, List[str]] = {
        'en': ['', 'Language: '], 'it': ['', 'Lingua: ']
    }
    LANGUAGE_INFO_FORMAT: Dict[str, List[str]] = {
        lang_id: [
            f' {{% if info and samples is any(sample => sample.language != (samples | first).language) %}}{l_par}{prefix}{language_format}{r_par}{{% endif %}}'
            for (l_par, r_par), prefix, language_format in product(
                [('(', ')'), ('[', ']')], LANGUAGE_INFO_PREFIX[lang_id], LANGUAGE_FORMAT
            )
        ]
        for lang_id in LANGUAGES
    }
    TRANSCRIPTION_PREFIX: Dict[str, Dict[str, List[str]]] = {
        'asr': {'en': ['Transcription', 'Transcript', 'Text'], 'it': ['Trascrizione', 'Trascritto', 'Testo']},
        'tts': {'en': ['Sentence', 'Transcription', 'Transcript', 'Text'],
                'it': ['Frase', 'Trascrizione', 'Trascritto', 'Testo']}
    }
    TRANSCRIPTION_START: Dict[str, Dict[str, List[str]]] = {
        task: {
            lang_id: [
                         f'{transcription_prefix}{{% if numbered and  approach == "paired" and samples | length > 1 %}} {{{{ loop.index }}}}{{% endif %}}{language_format}: '
                         for transcription_prefix, language_format in
                         product(TRANSCRIPTION_PREFIX[task][lang_id], LANGUAGE_INFO_FORMAT[lang_id])
                     ] + [
                         f'{transcription_prefix}{{% if numbered and approach == "paired" and samples | length > 1 %}} {{{{ loop.index }}}}{{% endif %}}:{language_format} '
                         for transcription_prefix, language_format in
                         product(TRANSCRIPTION_PREFIX[task][lang_id], LANGUAGE_INFO_FORMAT[lang_id])
                     ]
            for lang_id in LANGUAGES
        }
        for task in TRANSCRIPTION_PREFIX
    }
    TRANSCRIPTION_FORMAT: List[str] = [
        '{0}{1}{{{{ sample.sentence | trim }}}}{1}{0}'.format(boundary, "\n" if len(boundary) > 1 else "")
        for boundary in ['', '\'', '"', '\'\'\'', '"""']
    ]

    def __init__(
            self,
            *args,
            tokenizer: PreTrainedTokenizer,
            task_proportions: Optional[Dict[str, float]] = None,
            voices_mixing_proportions: Union[Dict[str, float], float] = 0.5,
            max_samples_per_task: int = 1,
            random_seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        #
        if task_proportions is not None:
            task_proportions = {k: v / sum(task_proportions.values()) for k, v in task_proportions.items()}
        else:
            task_proportions = dict(zip(self._tasks, (0.2, 0.2, 0.3, 0.3)))
        if isinstance(voices_mixing_proportions, float):
            voices_mixing_proportions = dict(zip(self._tasks, (voices_mixing_proportions,) * len(self._tasks)))
        #
        self.task_proportions: Dict[str, float] = task_proportions
        self.voices_mixing_proportions: Dict[str, float] = voices_mixing_proportions
        self.max_samples_per_task: int = max_samples_per_task
        self.random_seed: Optional[int] = random_seed
        #
        self.preprocessed_data = None
        self._prepare_samples()

    def _get_random_template(self, task: Literal['speak', 'rec', 'asr', 'tts'], lang_id: str) -> Template:
        #

        #
        if task == 'speak':
            template = self.SPEAK_TEMPLATE
        elif task == 'rec':
            (
                approach,
                instructions,
                samples_start,
                samples_sep,
                sample_io_sep,
                audio_start,
                audio_format,
                language_start,
                language_format
            ) = (
                random.choice(['paired', 'split']),
                random.choice(self.REC_INSTRUCTIONS[lang_id]),
                random.choice(self.SAMPLE_START),
                random.choice(self.SAMPLES_SEP),
                random.choice(self.SAMPLE_IO_SEP),
                random.choice(self.AUDIO_START[lang_id]),
                random.choice(self.AUDIO_FORMAT[lang_id]),
                random.choice(self.LANGUAGE_START[lang_id]),
                random.choice(self.LANGUAGE_FORMAT)
            )
            template = (
                f'{{% set approach = "{approach}" %}}'
                f'{{% set info = false %}}'
                f'{{% set numbered = false %}}'
                f'{instructions}{samples_start}'
                f'{{% if approach == "paired" %}}'
                f'{{% for sample in samples %}}{audio_start}{audio_format}{sample_io_sep}{language_start}{language_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% elif approach == "split" %}}'
                f'{{% for sample in samples %}}{audio_start}{audio_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}{sample_io_sep}{{% for sample in samples %}}{language_start}{language_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% endif %}}'
                f'{{{{tokenizer.eos_token}}}}'
            )
        elif task == 'asr':
            (
                approach,
                info,
                numbered,
                instructions,
                samples_start,
                samples_sep,
                sample_io_sep,
                audio_start,
                audio_format,
                transcription_start,
                transcription_format
            ) = (
                random.choice(['paired', 'split']),
                random.choice(['true', 'false']),
                random.choice(['true', 'false']),
                random.choice(self.REC_INSTRUCTIONS[lang_id]),
                random.choice(self.SAMPLE_START),
                random.choice(self.SAMPLES_SEP),
                random.choice(self.SAMPLE_IO_SEP),
                random.choice(self.AUDIO_START[lang_id]),
                random.choice(self.AUDIO_FORMAT[lang_id]),
                random.choice(self.TRANSCRIPTION_START['asr'][lang_id]),
                random.choice(self.TRANSCRIPTION_FORMAT)
            )
            template = (
                f'{{% set approach = "{approach}" %}}'
                f'{{% set info = {info} %}}'
                f'{{% set numbered = {numbered} %}}'
                f'{instructions}{samples_start}'
                f'{{% if approach == "paired" %}}'
                f'{{% for sample in samples %}}{audio_start}{audio_format}{sample_io_sep}{transcription_start}{transcription_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% elif approach == "split" %}}'
                f'{{% for sample in samples %}}{audio_start}{audio_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}{sample_io_sep}{{% for sample in samples %}}{transcription_start}{transcription_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% endif %}}'
                f'{{{{tokenizer.eos_token}}}}'
            )
        elif task == 'tts':
            (
                approach,
                info,
                numbered,
                instructions,
                samples_start,
                samples_sep,
                sample_io_sep,
                audio_start,
                audio_format,
                transcription_start,
                transcription_format
            ) = (
                random.choice(['paired', 'split']),
                random.choice(['true', 'false']),
                random.choice(['true', 'false']),
                random.choice(self.REC_INSTRUCTIONS[lang_id]),
                random.choice(self.SAMPLE_START),
                random.choice(self.SAMPLES_SEP),
                random.choice(self.SAMPLE_IO_SEP),
                random.choice(self.AUDIO_START[lang_id]),
                random.choice(self.AUDIO_FORMAT[lang_id]),
                random.choice(self.TRANSCRIPTION_START['tts'][lang_id]),
                random.choice(self.TRANSCRIPTION_FORMAT)
            )
            template = (
                f'{{% set approach = "{approach}" %}}'
                f'{{% set info = {info} %}}'
                f'{{% set numbered = {numbered} %}}'
                f'{instructions}{samples_start}'
                f'{{% if approach == "paired" %}}'
                f'{{% for sample in samples %}}{transcription_start}{transcription_format}{sample_io_sep}{audio_start}{audio_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% elif approach == "split" %}}'
                f'{{% for sample in samples %}}{transcription_start}{transcription_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}{sample_io_sep}{{% for sample in samples %}}{audio_start}{audio_format}{{% if not loop.last %}}{samples_sep}{{% endif %}}{{% endfor %}}'
                f'{{% endif %}}'
                f'{{{{tokenizer.eos_token}}}}'
            )
        else:
            raise ValueError()

        return self._jinja_env.from_string(template)

    def _prepare_samples(self):
        #
        self.data['task'] = None
        task_proportions = [*self.task_proportions.items()]
        rescaled_task_proportions = [
            (task, proportion / sum(v for _, v in task_proportions[i:]) if i > 0 else proportion)
            for i, (task, proportion) in enumerate(task_proportions)
        ]
        #
        for i, (task, proportion) in enumerate(rescaled_task_proportions):
            if proportion < 1:
                gss = GroupShuffleSplit(n_splits=1, train_size=proportion, random_state=self.random_seed)
                remaining_df = self.data[self.data['task'].apply(lambda x: x is None)]
                groups = remaining_df['client_id']
                idxs, _ = next(gss.split(remaining_df, groups=groups))
                self.data.loc[remaining_df.iloc[idxs].index, 'task'] = task
            else:
                self.data[self.data['task'].apply(lambda x: x is None)]['task'] = task
        #
        self.data['mixed_voice'] = False
        for task, proportion in self.voices_mixing_proportions.items():
            gss = GroupShuffleSplit(n_splits=1, train_size=proportion, random_state=self.random_seed)
            df = self.data[self.data['task'] == task]
            groups = df['client_id']
            idxs, _ = next(gss.split(df, groups=groups))
            self.data.loc[df.iloc[idxs].index, 'mixed_voice'] = True
        #
        self.data['group_id'] = 0
        for task in self._tasks:
            for mix in [True, False]:
                if mix:
                    for _, group in self.data[(self.data['task'] == task) & (self.data['mixed_voice'] == mix)].groupby('client_id'):
                        n = len(group)
                        offset = self.data['group_id'].max()
                        group_ids = np.arange(n) % int(n / self.max_samples_per_task) + offset
                        np.random.shuffle(group_ids)
                        self.data.loc[group.index, 'group_id'] = group_ids
                else:
                    n = len(self.data[(self.data['task'] == task) & (self.data['mixed_voice'] == mix)]['group_id'])
                    offset = self.data['group_id'].max()
                    group_ids = np.arange(n) % int(n / self.max_samples_per_task) + offset
                    np.random.shuffle(group_ids)
                    self.data[(self.data['task'] == task) & (self.data['mixed_voice'] == mix)]['group_id'] = group_ids
        #
        self.preprocessed_data = self.data.group_by('group_id')

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, index: int):
        samples = self.preprocessed_data.get_group(index)
        lang_id = random.choice(self.languages)
        task, *_ = samples['task'].unique()
        template = self._get_random_template(task, lang_id)

        return {
            'text': template.render(
                samples=[sample.to_dict() for sample in samples],
                languages=self._languages_dict[lang_id]
            ),
            'audio_file_paths': [os.path.join(sample['base_path'], 'clips', sample['path']) for sample in samples]
        }
