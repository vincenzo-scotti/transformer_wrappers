import unittest

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformer_wrappers.wrappers import ParallelTransformerWrapper, ParallelCausalLMWrapper

from .test_base import TestTransformerWrapper, TestCausalLMWrapper


class TestParallelModelWrapper(TestTransformerWrapper):
    transformer_wrapper = ParallelTransformerWrapper


class TestParallelCausalLMWrapper(TestCausalLMWrapper):
    causal_lm_wrapper = ParallelCausalLMWrapper


if __name__ == '__main__':
    unittest.main()
