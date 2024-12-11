from dataclasses import dataclass

from typing import List, Union, Any, Dict, Iterable, Optional

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers import PreTrainedTokenizer, BatchEncoding


@dataclass
class CausalLMDataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizer
    return_tensors: str = 'pt'

    def tf_call(self, *args, **kwargs):
        raise NotImplementedError()

    def np_call(self):
        raise NotImplementedError()

    def _prepare_input(self, text: Union[Iterable[str], str]) -> BatchEncoding:
        return self.tokenizer(text, return_tensors='pt', padding=True)

    def _prepare_output(
            self,
            text: Optional[Union[Iterable[str], str]] = None,
            input_data: Optional[BatchEncoding] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        if input_data is None:
            return self._prepare_output(input_data=self._prepare_input(text))
        else:
            output_ids = input_data.input_ids.clone()
        output_ids[~(input_data.attention_mask.bool())] = -100

        return output_ids

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> BatchEncoding:
        batch_encoding = self._prepare_input(sample['text'] for sample in examples)  # TODO fixme
        batch_encoding['labels'] = self._prepare_output(input_data=batch_encoding)

        return batch_encoding
