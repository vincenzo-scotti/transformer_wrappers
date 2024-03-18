import unittest
import os

import itertools

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformer_wrappers.wrappers import ParallelTransformerWrapper, ParallelCausalLMWrapper


class TestTransformerWrapper(unittest.TestCase):
    transformer_wrapper = ParallelTransformerWrapper

    def _test_forward(
            self,
            transformer,
            model_args=None,
            model_kwargs=None,
            tokenizer=None,
            tokenizer_args=None,
            tokenizer_kwargs=None,
            **wrapper_kwargs
    ):
        model_args = model_args if model_args is not None else tuple()
        model_kwargs = model_kwargs if model_kwargs is not None else dict()
        tokenizer = tokenizer if tokenizer is not None else transformer
        tokenizer_args = tokenizer_args if tokenizer_args is not None else tuple()
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else dict()

        input_string = 'Hello, World!\n'

        model = AutoModel.from_pretrained(transformer, *model_args, **model_kwargs)
        tok = AutoTokenizer.from_pretrained(tokenizer, *tokenizer_args, **tokenizer_kwargs)
        input_encodings = tok(input_string, return_tensors='pt').to(model.device)
        output = model(
            **input_encodings,
            return_dict=True,
            output_attentions=True,
            use_cache=True,
            output_hidden_states=True
        )

        model = self.transformer_wrapper.from_pretrained(
            transformer,
            model_args=model_args,
            model_kwargs=model_kwargs,
            tokenizer_name_or_path=tokenizer,
            tokenizer_args=tokenizer_args,
            tokenizer_kwargs=tokenizer_kwargs,
            **wrapper_kwargs
        )
        input_encodings = model.tokenizer(input_string, return_tensors='pt').to(model.device)
        output_wrapper = model(
            **input_encodings,
            return_dict=True,
            output_attentions=True,
            use_cache=True,
            output_hidden_states=True,
            return_attention_output=True,  # Self-attention layer output
            return_feed_forward_output=True
        )

        assert torch.equal(
            output.last_hidden_state, output_wrapper['output_hidden_state']
        ), '`last_hidden_state` not matching.'

        for i, (output_hidden_state, output_wrapper_hidden_state) in enumerate(zip(
                output.hidden_states, output_wrapper['hidden_states']
        )):
            if i == 0:
                assert torch.equal(
                    output_hidden_state, output_wrapper_hidden_state
                ), 'Initial embedding tensors not matching.'
            if i == len(model.layers):
                assert torch.equal(
                    output_hidden_state, model.norm(output_wrapper_hidden_state)
                ), f'`hidden_state` tensors at layer {i} not matching.'
            else:
                assert torch.equal(
                    output_hidden_state, output_wrapper_hidden_state
                ), f'`hidden_state` tensors at layer {i} not matching.'

        for i, (
                output_hidden_state, prev_output_wrapper_hidden_state, attn_output_wrapper, ffnn_output_wrapper
        ) in enumerate(zip(
            output.hidden_states[1:],
            output_wrapper['hidden_states'][:-1],
            output_wrapper['attention_outputs'],
            output_wrapper['feed_forward_outputs']
        ), start=1):
            output_wrapper_hidden_state = prev_output_wrapper_hidden_state + attn_output_wrapper + ffnn_output_wrapper
            if i == len(model.layers):
                assert torch.equal(
                    output_hidden_state, model.norm(output_wrapper_hidden_state)
                ), f'Composed `hidden_state` tensors at layer {i} not matching.'
            else:
                assert torch.equal(
                    output_hidden_state, output_wrapper_hidden_state
                ), f'Composed `hidden_state` tensors at layer {i} not matching.'

        for i, (output_past_key_values, output_wrapper_past_key_values) in enumerate(zip(
                output.past_key_values, output_wrapper['cache'],
        ), start=1):
            output_past_keys, output_past_values = output_past_key_values
            output_wrapper_past_keys, output_wrapper_past_values = output_wrapper_past_key_values
            assert torch.equal(
                output_past_keys, output_wrapper_past_keys
            ), f'`key` tensors at layer {i} not matching.'
            assert torch.equal(
                output_past_values, output_wrapper_past_values
            ), f'`value` tensors at layer {i} not matching.'

        for i, (output_attentions, output_wrapper_attentions) in enumerate(zip(
                output.attentions, output_wrapper['attention_weights']
        ), start=1):
            assert torch.equal(
                output_attentions, output_wrapper_attentions
            ), f'`attentions` tensors at layer {i} not matching.'

    def _test_forward_configs(self, *args, **kwargs):
        for block_parallel, iterative, scaling in itertools.product([True, False], [True, False], [True, False]):
            self._test_forward(*args, **kwargs, block_parallel=block_parallel, iterative=iterative, scaling=scaling)

    def test_gpt2_forward(self):
        self._test_forward_configs('gpt2')

    def test_mistral_forward(self):
        self._test_forward_configs(
            'mistralai/Mistral-7B-Instruct-v0.2',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }
        )

    def test_gemma_forward(self):
        self._test_forward_configs(
            'google/gemma-7b',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )

    def test_llama2_forward(self):
        self._test_forward_configs(
            'meta-llama/Llama-2-7b-hf',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )


class TestCausalLMWrapper(unittest.TestCase):
    causal_lm_wrapper = ParallelCausalLMWrapper

    def _test_forward(
            self,
            transformer,
            model_args=None,
            model_kwargs=None,
            tokenizer=None,
            tokenizer_args=None,
            tokenizer_kwargs=None,
            **wrapper_kwargs
    ):
        model_args = model_args if model_args is not None else tuple()
        model_kwargs = model_kwargs if model_kwargs is not None else dict()
        tokenizer = tokenizer if tokenizer is not None else transformer
        tokenizer_args = tokenizer_args if tokenizer_args is not None else tuple()
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else dict()

        input_string = 'Hello, World!\n'

        model = AutoModelForCausalLM.from_pretrained(transformer, *model_args, **model_kwargs)
        tok = AutoTokenizer.from_pretrained(tokenizer, *tokenizer_args, **tokenizer_kwargs)
        input_encodings = tok(input_string, return_tensors='pt').to(model.device)
        output = model.forward(
            **input_encodings,
            return_dict=True,
            output_attentions=True,
            use_cache=True,
            output_hidden_states=True
        )

        model = self.causal_lm_wrapper.from_pretrained(
            transformer,
            model_args=model_args,
            model_kwargs=model_kwargs,
            tokenizer_name_or_path=tokenizer,
            tokenizer_args=tokenizer_args,
            tokenizer_kwargs=tokenizer_kwargs,
            **wrapper_kwargs
        )
        input_encodings = model.tokenizer(input_string, return_tensors='pt').to(model.device)
        output_wrapper = model.forward(
            **input_encodings,
            return_dict=True,
            output_attentions=True,
            use_cache=True,
            output_hidden_states=True,
            return_attention_output=True,  # Self-attention layer output
            return_feed_forward_output=True
        )

        assert torch.equal(output.logits, output_wrapper['logits']), 'Logit tensors do not match.'

    def _test_forward_configs(self, *args, **kwargs):
        for block_parallel, iterative, scaling in itertools.product([True, False], [True, False], [True, False]):
            self._test_forward(*args, **kwargs, block_parallel=block_parallel, iterative=iterative, scaling=scaling)

    def test_gpt2_forward(self):
        self._test_forward_configs('gpt2')

    def test_mistral_forward(self):
        self._test_forward_configs(
            'mistralai/Mistral-7B-Instruct-v0.2',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }
        )

    def test_gemma_forward(self):
        self._test_forward_configs(
            'google/gemma-7b',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )

    def test_llama2_forward(self):
        self._test_forward_configs(
            'meta-llama/Llama-2-7b-hf',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )

    def _test_generate(
            self,
            transformer,
            model_args=None,
            model_kwargs=None,
            tokenizer=None,
            tokenizer_args=None,
            tokenizer_kwargs=None,
            **wrapper_kwargs
    ):
        model_args = model_args if model_args is not None else tuple()
        model_kwargs = model_kwargs if model_kwargs is not None else dict()
        tokenizer = tokenizer if tokenizer is not None else transformer
        tokenizer_args = tokenizer_args if tokenizer_args is not None else tuple()
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else dict()

        input_string = 'Hello, World!\n'

        model = AutoModelForCausalLM.from_pretrained(transformer, *model_args, **model_kwargs)
        tok = AutoTokenizer.from_pretrained(tokenizer, *tokenizer_args, **tokenizer_kwargs)
        input_encodings = tok(input_string, return_tensors='pt').to(model.device)
        output = model.generate(input_encodings.input_ids, do_sample=False, max_length=16)

        model = self.causal_lm_wrapper.from_pretrained(
            transformer,
            model_args=model_args,
            model_kwargs=model_kwargs,
            tokenizer_name_or_path=tokenizer,
            tokenizer_args=tokenizer_args,
            tokenizer_kwargs=tokenizer_kwargs,
            **wrapper_kwargs
        )
        input_encodings = model.tokenizer(input_string, return_tensors='pt').to(model.device)
        output_wrapper = model.generate(
            input_encodings.input_ids, return_inner_states=True, do_sample=False, max_length=16
        )

        assert torch.equal(output, output_wrapper['output_ids']), 'Generated token tensors do not match.'

    def _test_generate_configs(self, *args, **kwargs):
        for block_parallel, iterative, scaling in itertools.product([True, False], [True, False], [True, False]):
            self._test_generate(*args, **kwargs, block_parallel=block_parallel, iterative=iterative, scaling=scaling)

    def test_gpt2_generate(self):
        self._test_generate_configs('gpt2')

    def test_mistral_generate(self):
        self._test_generate_configs(
            'mistralai/Mistral-7B-Instruct-v0.2',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }
        )

    def test_gemma_generate(self):
        self._test_generate_configs(
            'google/gemma-7b',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )

    def test_llama2_generate(self):
        self._test_generate_configs(
            'meta-llama/Llama-2-7b-hf',
            model_kwargs={
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'eager',
                'device_map': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'token': os.environ['HUGGING_FACE_TOKEN']
            },
            tokenizer_kwargs={'token': os.environ['HUGGING_FACE_TOKEN']}
        )


if __name__ == '__main__':
    unittest.main()
