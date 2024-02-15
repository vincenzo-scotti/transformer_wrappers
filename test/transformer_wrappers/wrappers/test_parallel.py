import unittest

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformer_wrappers.wrappers import ParallelModelWrapper, ParallelModelWrapperForCausalLMWrapper


class TestParallelModelWrapper(unittest.TestCase):
    def test_gpt2_forward_no_parallel(self):
        transformer = 'gpt2'
        input_string = 'Hello, World!\n'

        model = AutoModel.from_pretrained(transformer)
        tokenizer = AutoTokenizer.from_pretrained(transformer)
        input_encodings = tokenizer(input_string, return_tensors='pt')
        output = model(
            **input_encodings, return_dict=True, output_attentions=True, use_cache=True, output_hidden_states=True
        )

        model = ParallelModelWrapper.from_pretrained(transformer)
        input_encodings = model.tokenizer(input_string, return_tensors='pt')
        output_wrapper = model(
            **input_encodings, return_dict=True, output_attentions=True, use_cache=True, output_hidden_states=True
        )

        assert torch.equal(
            output.last_hidden_state, output_wrapper.last_hidden_state
        ), '`last_hidden_state` not matching.'
        for i, (output_hidden_state, output_wrapper_hidden_state) in enumerate(zip(
                output.hidden_states, output_wrapper.hidden_states
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
        for i, (output_past_key_values, output_wrapper_past_key_values) in enumerate(zip(
                output.past_key_values, output_wrapper.past_key_values
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
                output.attentions, output_wrapper.attentions
        ), start=1):
            assert torch.equal(
                output_attentions, output_wrapper_attentions
            ), f'`attentions` tensors at layer {i} not matching.'


class TestParallelModelWrapperForCausalLMWrapper(unittest.TestCase):
    def test_gpt2_generate_no_parallel(self):
        transformer = 'gpt2'
        input_string = 'Hello, World!\n'

        model = AutoModelForCausalLM.from_pretrained(transformer)
        tokenizer = AutoTokenizer.from_pretrained(transformer)
        input_encodings = tokenizer(input_string, return_tensors='pt')
        output = model.generate(input_encodings.input_ids, do_sample=False, max_length=16)

        model = ParallelModelWrapperForCausalLMWrapper.from_pretrained(transformer)
        input_encodings = model.tokenizer(input_string, return_tensors='pt')
        output_wrapper = model.generate(input_encodings.input_ids, do_sample=False, max_length=16)

        assert torch.equal(output, output_wrapper), 'Generated token tensors do not match.'


if __name__ == '__main__':
    unittest.main()
