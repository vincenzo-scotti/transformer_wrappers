from typing import Union, List, Optional

import torch

from bigbench.api.model import Model, ModelData
from bigbench.models import model_utils

from transformer_wrappers.wrappers import TransformerWrapper, CausalLMWrapper
from transformer_wrappers.wrappers.constants import *


class BigBenchModel(Model):
    # Code adapted from https://github.com/google/BIG-bench/blob/main/bigbench/models/huggingface_models.py
    def __init__(
            self,
            model: CausalLMWrapper,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        self._model: CausalLMWrapper = model
        self._device: Optional[torch.device] = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._parameters = kwargs

    def generate_text(
            self,
            inputs: Union[str, List[str]],
            max_length: int,
            stop_string: Optional[str] = None,
            output_regex: Optional[str] = None,
            **kwargs
    ) -> Union[str, List[str]]:
        # Preprocess input
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs
        # Prepare parameters
        generate_kwargs = {**kwargs, **self._parameters}
        # Prepare output accumulator
        generated = list()
        # Iterate through inputs
        # TODO Add TQDM iterator
        for input_text in input_list:
            # Encode input
            input_encodings = self._model.tokenizer(input_text, return_tensors='pt').to(self._device)
            # Generate text
            generator_output = self._model.generate(input_encodings.input_ids, **generate_kwargs)
            output_text = self._model.tokenizer.decode(
                generator_output[OUTPUT_IDS][0, input_encodings.input_ids.size(1):], skip_special_tokens=True
            )
            # Accumulate output
            generated.append(output_text)
        # Match original input shape
        if isinstance(inputs, str):
            generated, *_ = generated
        # Apply final postprocessing
        generated = model_utils.postprocess_output(generated, max_length, stop_string, output_regex)

        return generated

    def cond_log_prob(
            self,
            inputs: Union[str, List[str]],
            targets: Union[List[str], List[List[str]]],
            absolute_normalization: Optional[bool] = False,
            batch_size: int = 1,  # TODO Use batch size
            # TODO parametrise what to score (all seq, first token, first different token)
            **kwargs
    ) -> Union[List[float], List[List[float]]]:
        # Preprocess input
        if isinstance(inputs, str):
            input_list = [inputs]
            target_list = [targets]
        else:
            input_list = inputs
            target_list = targets
        # Prepare output accumulator
        scores = list()
        # Iterate through inputs
        # TODO Add TQDM iterator
        for prompt, completions in zip(input_list, target_list):
            # Encode prompt
            prompt = prompt.strip()  # Make sure no spaces
            prompt_encodings = self._model.tokenizer(prompt, return_tensors='pt').to(self._device)
            # Process prompt and gather cache
            past_key_values = self._model.transformer_wrapper.forward(
                **prompt_encodings, use_cache=True, **kwargs
            )[CACHE]
            # Prepare scores accumulator
            completion_scores = list()
            # Iterate through completions
            for completion in completions:
                # Encode Completion
                completion = f'{completion.strip()}'  # Make sure no spaces
                completion_encodings = self._model.tokenizer(completion, return_tensors='pt').to(self._device)
                # Score completion
                logits = self._model.transformer_wrapper.forward(
                    **prompt_encodings, past_key_values=past_key_values, use_cache=True, **kwargs
                )[LOGITS]
                shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                shift_labels = completion_encodings.input_ids[..., 1:].contiguous().view(-1)
                score = - torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='sum')  # TODO change to first (?)
                # Accumulate output
                completion_scores = score
            # Accumulate output
            scores.append(completion_scores)
        # Match original input shape
        if isinstance(inputs, str):
            scores, *_ = scores

        return scores

    def model_data(self, *args, **kwargs):
        return ModelData(
            model_family='colab',
            model_name='colab',
            total_params=2,
            non_embedding_params=1,
            flop_matched_non_embedding_params=1,
            training_batch_size=1,
            training_steps=1,
            description='Dummy model for testing',
            decoding_params={}
        )
