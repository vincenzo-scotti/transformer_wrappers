import torch
import torch.nn.functional as F
from torchmetrics import Metric
from parlai.core.metrics import BleuMetric, F1Metric, IntraDistinctMetric, InterDistinctMetric

from collections import Counter

from typing import Optional, List, Literal

# NOTE there are pre-implemented versions of these metrics in torchmetrics

__all__ = ['PPLScore', 'BLEUScore', 'F1Score', 'DistinctNScore']


# PPL
class PPLScore(Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, ignore_idx: int = -100, **kwargs):
        super(PPLScore, self).__init__(**kwargs)
        self.ignore_idx: int = ignore_idx
        self.add_state("total_ppl_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @classmethod
    def compute_ppl_score(
            cls, logits: torch.tensor, labels: torch.tensor, ignore_idx: int = -100
    ) -> torch.tensor:
        # Compute NLL
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_idx, reduction='none'
        ).view(labels.size()).sum(dim=1)
        # Compute number of tokens
        n = (labels != ignore_idx).sum(dim=1)
        #
        ppl = torch.exp(nll / n)

        return ppl

    def update(self, preds: torch.tensor, targets: torch.tensor):
        assert len(preds) == len(targets)

        self.total_ppl_score += self.compute_ppl_score(preds, targets, self.ignore_idx).sum()
        self.count += len(targets)

    def compute(self):
        return self.total_ppl_score.float() / self.count


# BLEU
class BLEUScore(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, n_gram_size: int = 1, **kwargs):
        super(BLEUScore, self).__init__(**kwargs)
        self.n_gram_size: int = n_gram_size
        self.add_state("total_bleu_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: List[str], targets: List[str]):
        assert len(preds) == len(targets)

        self.total_bleu_score += sum(
            BleuMetric.compute(pred, [tgt], k=self.n_gram_size).value() for pred, tgt in zip(preds, targets)
        )
        self.count += len(targets)

    def compute(self):
        return self.total_bleu_score.float() / self.count


# F1
class F1Score(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super(F1Score, self).__init__(**kwargs)
        self.add_state("total_f1_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: List[str], targets: List[str]):
        assert len(preds) == len(targets)

        self.total_f1_score += sum(F1Metric.compute(pred, [tgt]).value() for pred, tgt in zip(preds, targets))
        self.count += len(targets)

    def compute(self):
        return self.total_f1_score.float() / self.count


# Distinct
class DistinctNScore(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, normalisation: Literal['seq', 'corpus'] = 'seq', n_gram_size: int = 1, **kwargs):
        super(DistinctNScore, self).__init__(**kwargs)
        self.normalisation: Literal['seq', 'corpus'] = normalisation
        self.n_gram_size: int = n_gram_size
        if self.normalisation == 'seq':
            self.add_state("total_distinct_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        elif self.normalisation == 'corpus':
            self.total_distinct_score = InterDistinctMetric(Counter())
        else:
            raise ValueError(f'Unknown normalisation approach: \'{self.normalisation}\'')

    def update(self, preds: List[str], *_):
        if self.normalisation == 'seq':
            self.total_distinct_score += sum(
                IntraDistinctMetric.compute(pred, ngram=self.n_gram_size).value() for pred in preds
            )
            self.count += len(preds)
        elif self.normalisation == 'corpus':
            for pred in preds:
                self.total_distinct_score += InterDistinctMetric.compute(pred, ngram=self.n_gram_size)
        else:
            raise ValueError(f'Unknown normalisation approach: \'{self.normalisation}\'')

    def compute(self):
        if self.normalisation == 'seq':
            return self.total_distinct_score.float() / self.count
        elif self.normalisation == 'corpus':
            return torch.tensor(self.total_distinct_score.value())
        else:
            raise ValueError(f'Unknown normalisation approach: \'{self.normalisation}\'')

    def reset(self):
        super(DistinctNScore, self).reset()
        if self.normalisation == 'corpus':
            self.total_distinct_score = InterDistinctMetric(Counter())
