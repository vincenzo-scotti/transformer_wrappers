import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import torchmetrics

from transformers import AutoTokenizer, AutoConfig
from transformers import logging

from typing import Tuple, Optional, Dict, Set, Union, Iterable, Literal, List, Pattern

from .base import TransformerWrapper


__all__ = ['CharTokenizer', 'TokeNN']


logger = logging.get_logger(__name__)


class CharTokenizer:
    escaped_tokens_regex: Pattern[str] = re.compile(r'<0x(\w\w)>')

    def __init__(self, vocabulary: Set[str], special_tokens_map: Dict[str, str]):
        #
        self.vocabulary: Set[str] = vocabulary
        self.special_tokens_map: Dict[str, str] = special_tokens_map
        self.decoder: Tuple[str, ...] = tuple(set(self.special_tokens_map.values())) + tuple(self.vocabulary)
        self.encoder: Dict[str, int] = dict(zip(self.decoder, range(len(self.decoder))))
        #
        special_tokens_pattern = '|'.join(re.escape(s) for s in set(special_tokens_map.values()))
        tokens_pattern = f'({special_tokens_pattern}|.{{1}}|\n{{1}})'
        self.tokenizer_regex = re.compile(tokens_pattern, flags=re.UNICODE)

    @property
    def bos_token(self):
        return self.special_tokens_map.get('bos_token')

    @property
    def bos_token_id(self):
        return self.encoder.get(self.bos_token)

    @property
    def eos_token(self):
        return self.special_tokens_map.get('eos_token')

    @property
    def eos_token_id(self):
        return self.encoder.get(self.eos_token)

    @property
    def pad_token(self):
        return self.special_tokens_map.get('pad_token', self.eos_token)

    @property
    def pad_token_id(self):
        return self.encoder.get(self.pad_token)

    @property
    def unk_token(self):
        return self.special_tokens_map.get('unk_token')

    @property
    def unk_token_id(self):
        return self.encoder.get(self.unk_token)

    @property
    def cls_token(self):
        return self.special_tokens_map.get('cls_token')

    @property
    def cls_token_id(self):
        return self.encoder.get(self.cls_token)

    def __len__(self):
        return len(self.vocabulary) + len(set(self.special_tokens_map.values()))

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(
            self,
            text: Union[Iterable[str], str],
            return_tensors: Union[bool, Literal['pt', 'np']] = False,
            padding: Union[bool, Literal['truncate', 'max_length']] = True,
            device: Optional[torch.device] = None
    ) -> Union[Dict[str, List[int]], Dict[str, List[List[int]]], Dict[str, np.ndarray], Dict[str, torch.tensor]]:
        if isinstance(text, str):
            text = text.replace(' ', '▁')
            tokens = self.tokenizer_regex.findall(text)

            ids = [self.encoder.get(token, self.unk_token_id) for token in tokens]
            valid_mask = [1] * len(ids)
        else:
            ids, valid_mask = list(zip(*[self.encode(text_).values() for text_ in text]))
            ids = list(ids)
            valid_mask = list(valid_mask)

        if padding and not isinstance(text, str):
            seq_lengths = [len(elem) for elem in ids]

            if padding is True or padding == 'max_length':
                max_seq_length = max(seq_lengths)
                padding_id = self.encoder[self.special_tokens_map.get('pad_token', self.special_tokens_map['eos_token'])]
                ids = [ids_ + [padding_id] * (max_seq_length - seq_len) for ids_, seq_len in zip(ids, seq_lengths)]
                valid_mask = [mask + [0] * (max_seq_length - seq_len) for mask, seq_len in zip(valid_mask, seq_lengths)]
            elif padding == 'truncate':
                min_seq_length = min(seq_lengths)
                ids = [ids_[:min_seq_length] for ids_ in ids]
                valid_mask = [mask[:min_seq_length] for mask in valid_mask]
            else:
                raise ValueError(f'Unsupported padding type: {padding}')

        if return_tensors:
            assert padding, 'Padding is required for returning tensors.'

            if isinstance(text, str):
                ids = [ids]
                valid_mask = [valid_mask]

            if return_tensors is True or return_tensors == 'pt':
                ids = torch.LongTensor(ids, device=device)
                valid_mask = torch.LongTensor(valid_mask, device=device)
            elif return_tensors == 'np':
                ids = np.array(ids)
                valid_mask = np.array(valid_mask)

        return {'input_ids': ids, 'valid_mask': valid_mask}

    def decode(
            self,
            ids: Union[torch.tensor, np.ndarray, Iterable[Iterable[int]], Iterable[int], int],
            skip_special_tokens: bool = False
    ) -> Union[List[str], str]:
        if isinstance(ids, int):
            return self.decoder[ids] if ids not in self.special_tokens_map.values() else ''
        elif isinstance(ids, torch.Tensor):
            return self.decode(ids.cpu().tolist(), skip_special_tokens=skip_special_tokens)
        elif isinstance(ids, np.ndarray):
            return self.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
        elif isinstance(ids, Iterable) and all(isinstance(elem, int) for elem in ids):
            return ''.join(
                self.decoder[id_] for id_ in ids
                if ids not in self.special_tokens_map.values() or not skip_special_tokens
            ).replace('▁', ' ')
        else:
            return [self.decode(ids_, skip_special_tokens=skip_special_tokens) for ids_ in ids]

    def get_out_gate(
            self,
            tokens: Union[Iterable[Iterable[str]], Iterable[str], str],
            return_tensors: Union[bool, Literal['pt', 'np']] = False,
            padding: Union[bool, Literal['truncate', 'max_length']] = True,
            device: Optional[torch.device] = None
    ) -> Union[List[float], List[List[float]], np.ndarray, torch.tensor]:
        if isinstance(tokens, str):
            out_gate = [0.] * len(self.encode(tokens)['input_ids'])
            out_gate[-1] = 1.
        elif isinstance(tokens, Iterable) and all(isinstance(token, str) for token in tokens):
            out_gate = sum((self.get_out_gate(token) for token in tokens), list())
        else:
            out_gate = [self.get_out_gate(tokens_) for tokens_ in tokens]

        if padding and not (
            isinstance(tokens, str) or (
                isinstance(tokens, Iterable) and all(isinstance(token, str) for token in tokens)
            )
        ):
            seq_lengths = [len(elem) for elem in out_gate]

            if padding is True or padding == 'max_length':
                max_seq_length = max(seq_lengths)
                out_gate = [
                    out_gate_ + [0.] * (max_seq_length - seq_len) for out_gate_, seq_len in zip(out_gate, seq_lengths)
                ]
            elif padding == 'truncate':
                min_seq_length = min(seq_lengths)
                out_gate = [out_gate_[:min_seq_length] for out_gate_ in out_gate]
            else:
                raise ValueError(f'Unsupported padding type: {padding}')

        if return_tensors:
            assert padding, 'Padding is required for returning tensors.'

            if isinstance(tokens, str) or (
                isinstance(tokens, Iterable) and all(isinstance(token, str) for token in tokens)
            ):
                out_gate = [out_gate]

            if return_tensors is True or return_tensors == 'pt':
                out_gate = torch.FloatTensor(out_gate, device=device)
            elif return_tensors == 'np':
                out_gate = np.array(out_gate)

        return out_gate


class TokeNN(L.LightningModule):
    # TODO create custom tokeniser class
    def __init__(
            self,
            vocabulary: Set[str],
            special_tokens_map: Dict[str, str],
            hidden_size: int,
            **kwargs
    ):
        super(TokeNN, self).__init__()
        #
        self.char_tokenizer = CharTokenizer(vocabulary, special_tokens_map)
        #
        self.embedding = nn.Embedding(len(self.char_tokenizer), hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        #
        self.metrics: Optional[Dict[str, torchmetrics.Metric]] = None
        #
        self.additional_parameters: Dict = kwargs

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            weights_path: Optional[Union[str, os.PathLike]] = None,
            init_embeddings: bool = False,
            **kwargs
    ):
        assert weights_path is None or init_embeddings is None

        model_args = kwargs.pop('model_args', tuple())
        model_kwargs = kwargs.pop('model_kwargs', dict())
        tokenizer_args = kwargs.pop('tokenizer_args', tuple())
        tokenizer_kwargs = kwargs.pop('tokenizer_kwargs', dict())

        configs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, *model_args, **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *tokenizer_args, **tokenizer_kwargs
        )
        tokenn = cls(
            set(
                [tok for tok in tokenizer.vocab if len(tok) == 1] +
                [
                    chr(int(CharTokenizer.escaped_tokens_regex.match(tok)[1], 16))
                    for tok in tokenizer.vocab
                    if CharTokenizer.escaped_tokens_regex.match(tok)
                ]
            ),  # ,
            tokenizer.special_tokens_map,
            configs.hidden_size,
            **kwargs
        )
        if weights_path is not None:
            tokenn.load_state_dict(torch.load(weights_path)['state_dict'])
        else:
            logger.warning(
                'The loaded TokeNN model has not been trained yet, you must train it before use.'
            )
        if init_embeddings:
            initial_embeddings = TransformerWrapper.from_pretrained(
                pretrained_model_name_or_path,
                model_args=model_args,
                model_kwargs=model_kwargs,
                tokenizer_args=tokenizer_args,
                tokenizer_kwargs=tokenizer_kwargs
            ).embedding
            with torch.no_grad():
                tokenn.embedding.weight[:] = initial_embeddings(
                    torch.tensor(tokenizer.convert_tokens_to_ids(tokenn.char_tokenizer.decoder))
                )

        return tokenn

    def configure_optimizers(self):
        optimizer_parameters = {
            k.split('__', 0)[1]: v for k, v in self.additional_parameters.items() if k.startswith('optimizer__')
        }
        optimiser = torch.optim.AdamW(self.parameters(), **optimizer_parameters)

        return optimiser

    def configure_metrics(self):
        self.metrics = {
            'Accuracy': torchmetrics.Accuracy('binary'),
            'Precision': torchmetrics.Precision('binary'),
            'Recall': torchmetrics.Recall('binary'),
            'Specificity': torchmetrics.Specificity('binary'),
            'F1': torchmetrics.F1Score('binary'),
            'Calibration Error': torchmetrics.CalibrationError('binary')
        }

    def forward(
            self,
            input_ids: torch.tensor,
            valid_mask: Optional[torch.tensor] = None,
            out_gate: Optional[torch.tensor] = None,
            return_attention_mask: bool = False
    ) -> Tuple[torch.tensor, torch.tensor]:
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
        else:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if out_gate is not None:
            out_gate = out_gate.bool()

        h: torch.tensor = self.embedding(input_ids)

        out_gate_logits = list()
        embeddings = [list() for _ in range(input_ids.size(0))]
        for i in range(h.size(1)):
            if i > 0:
                mask = ~(out_gate[:, i - 1] if out_gate is not None else out_gate_logits[i - 1] > 0) & valid_mask[:, i]
                h[mask, i] += h[mask, i - 1]
            out_gate_logits.append(self.output(h[:, i]).squeeze(-1))
            if i < h.size(1) - 1:
                out_flags = (
                    (out_gate[:, i] if out_gate is not None else out_gate_logits[i] > 0) & valid_mask[:, i]
                ) | (
                    valid_mask[:, i] & ~valid_mask[:, i + 1]
                )
            else:
                out_flags = valid_mask[:, i]
            embeddings = [
                embeddings_ + ([curr_embedding.clone()] if out_flag > 0 else list())
                for embeddings_, curr_embedding, out_flag in zip(embeddings, h[:, i], out_flags)
            ]

        out_gate_logits = torch.stack(out_gate_logits, dim=1)
        seq_lengths = [len(elem) for elem in embeddings]
        max_seq_length = max(seq_lengths)
        embeddings = [
            F.pad(
                torch.stack(embeddings_),
                (0, 0, 0, max_seq_length - seq_length)
            )
            for embeddings_, seq_length in zip(embeddings, seq_lengths)
            if seq_length > 0
        ]
        embeddings = torch.stack(embeddings)

        if return_attention_mask:
            attention_mask = torch.ones(
                len(embeddings), max_seq_length, dtype=torch.int, device=embeddings.device
            )
            for i, seq_length in enumerate(seq_lengths):
                attention_mask[i, seq_length:] = 0

            return embeddings, attention_mask
        else:
            return embeddings, out_gate_logits

    def _step(
            self,
            split,
            mini_batch: Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
            mini_batch_idx: int
    ):
        # Unpack mini-batch
        input_ids, valid_mask, tgt_embeddings, tgt_out_gate, tgt_attention_mask = mini_batch
        # Compute output
        outputs = self.forward(input_ids, valid_mask=valid_mask, out_gate=tgt_out_gate)
        embeddings, out_gate_logits = outputs
        # Compute loss
        embedding_loss = F.mse_loss(
            torch.masked_select(embeddings, tgt_attention_mask.bool().unsqueeze(-1)).view(-1, embeddings.size(-1)),
            torch.masked_select(tgt_embeddings, tgt_attention_mask.bool().unsqueeze(-1)).view(-1, tgt_embeddings.size(-1))
        )
        out_gate_loss = F.binary_cross_entropy_with_logits(out_gate_logits.reshape(-1), tgt_out_gate.reshape(-1))
        loss = embedding_loss + out_gate_loss
        # Log losses
        self.log(f'Embedding MSE/{split}', embedding_loss)
        self.log(f'Output Gate BCE/{split}', out_gate_loss)
        self.log(f'Loss/{split}', loss)

        return outputs, loss

    def training_step(self, *args, **kwargs) -> torch.tensor:
        _, loss = self._step('Training', *args, **kwargs)

        return loss

    def _eval_step(
            self,
            split,
            mini_batch: Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
            mini_batch_idx: int
    ) -> torch.tensor:
        # Unpack mini-batch
        *_, tgt_out_gate, _ = mini_batch
        # Compute outputs
        (_, out_gate_logits), loss = self._step(split, mini_batch, mini_batch_idx)
        # Compute metrics
        for metric in self.metrics.values():
            metric.update(out_gate_logits.reshape(-1), tgt_out_gate.reshape(-1))

        return loss

    def validation_step(self, mini_batch, mini_batch_idx):
        return self._eval_step('Validation', mini_batch, mini_batch_idx)

    def test_step(self, mini_batch, mini_batch_idx):
        return self._eval_step('Test', mini_batch, mini_batch_idx)

    def _evaluation_epoch_start(self):
        for metric in self.metrics.values():
            metric.reset()

    def on_validation_epoch_start(self):
        return self._evaluation_epoch_start()

    def on_test_epoch_start(self):
        return self._evaluation_epoch_start()

    def _evaluation_epoch_end(self, split: str):
        for metric_id, metric in self._lm_metrics.items():
            self.log(f'{metric_id}/{split}', metric.compute())

    def on_validation_epoch_end(self):
        return self._evaluation_epoch_end('Validation')

    def on_test_epoch_end(self):
        return self._evaluation_epoch_end('Test')
