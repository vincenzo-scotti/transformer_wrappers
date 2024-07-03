from inspect import signature, Parameter
from typing import List, Set
from copy import copy

import inspect

from .base import  CausalLMWrapper
from .constants import * # pylint:disable=W0401,W0614


class ParameterCausalLMWrapper(CausalLMWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_parameters: Set[str] = set()
        self._original_parameters: List[Parameter] = list(signature(self.base_model.forward).parameters.values())

    @property
    def gen_parameters(self):
        return copy(self._gen_parameters)

    def add_parameters(self, parameters: List[str] = None):
        if parameters:
            self._gen_parameters = self._gen_parameters.union(set(parameters))
            new_parameters = self._original_parameters + [
                Parameter(param, Parameter.KEYWORD_ONLY, default=None) for param in self._gen_parameters]
            self.base_model.forward.__signature__ = signature(
                self.base_model.forward).replace(
                parameters=new_parameters)

    def remove_parameters(self, parameters: List[str] = None):
        if parameters:
            self._gen_parameters = self._gen_parameters.difference(set(parameters))
            new_parameters = self._original_parameters + [
                Parameter(param, Parameter.KEYWORD_ONLY, default=None) for param in self._gen_parameters]
            self.base_model.forward.__signature__ = signature(
                self.base_model.forward).replace(
                parameters=new_parameters)

    def prepare_inputs_for_generation(self, *args, base_model_output: bool = True, **kwargs):
        #
        inputs = self._model.prepare_inputs_for_generation(*args, **kwargs)
        if self.is_wrapping:
            inputs |= {"base_model_output": base_model_output}
        # Add custom generation parameters
        inputs |= {param: kwargs[param] for param in self._gen_parameters if param in kwargs}

        return inputs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        if not self.is_wrapping:
            return self.base_model.generate(*args, **kwargs)

        # Bypass CausalLMWrapper's generate implementation
        generate_output = super(CausalLMWrapper, self).generate(*args, **kwargs) # pylint:disable=E1101

        # Re-run through layers to collect all data  # TODO find better solution
        if return_inner_states or not self.is_benchmarking:
            #
            return self.forward(
                input_ids=generate_output,
                **{
                    k: kwargs.get(k)
                    for k in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())
                    if k not in {"args", "kwargs", "self", "base_model_output"}
                } | {param: kwargs[param] for param in self._gen_parameters if param in kwargs},
                return_dict=True,
                output_attentions=True,
                use_cache=True,
                output_hidden_states=True,
                return_attention_output=True,  # Self-attention layer output
                return_feed_forward_output=True,
                return_intermediate_hidden_states=True
            ) | {"output_ids": generate_output}
        return generate_output
