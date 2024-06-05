import sys
from argparse import ArgumentParser, Namespace
import logging
import os
import hashlib

import json

from transformer_wrappers.utils.scripting import init_evaluation_environment
from transformer_wrappers.wrappers import CausalLMWrapper, causal_lm_mapping

import lm_eval
from lm_eval.utils import handle_non_serializable

from typing import Dict, Type


def get_configs_hash(configs):
    return hashlib.sha256(str(configs).encode()).hexdigest()


def main(args: Namespace):
    # Init environment
    configs: Dict = init_evaluation_environment(args.config_file_path)
    # Start logging info
    logging.info("Script started and configuration file loaded")
    # Build model
    model_type: Type[CausalLMWrapper] = causal_lm_mapping[configs['model'].pop('dtype')]
    model: CausalLMWrapper = model_type(**configs['model'])
    logging.info("Model built")
    # Run evaluations
    model.enable_benchmarking()
    for exp_config in configs['params']:
        #
        exp_config_hash = get_configs_hash(exp_config)
        results_dump_path = os.path.join(
            os.path.join(configs['current_experiment_dir_path'],  f'results_{exp_config_hash}.json')
        )
        configs_dump_path = os.path.join(
            os.path.join(configs['current_experiment_dir_path'],  f'configs_{exp_config_hash}.json')
        )
        #
        if not (os.path.exists(results_dump_path) and os.path.exists(configs_dump_path)) or configs['overwrite']:
            # Set attn len
            for k, v in exp_config.items():
                setattr(model.transformer_wrapper, k, v)
            # Run evaluation
            results = lm_eval.simple_evaluate(
                model='hf',
                model_args={'pretrained': model, 'tokenizer': model.tokenizer, 'backend': 'causal'},
                **configs['lm_eval']
            )
            # Save results
            if results is not None:
                with open(results_dump_path, 'w') as f:
                    json.dump(
                        results, f, indent=2, default=handle_non_serializable, ensure_ascii=False
                    )
                logging.info("Results serialised")
                with open(configs_dump_path, 'w') as f:
                    json.dump(
                        exp_config, f, indent=2, default=handle_non_serializable, ensure_ascii=False
                    )
                logging.info("Configs serialised")
            else:
                logging.error(f"Null results for configuration: `{exp_config}`")
    # Close script info
    logging.info("Script executed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='fit_eval_script',
        description='Script evaluate a causal transformer LM wrapper with the LM evaluation harness tool'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the evaluation configurations."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
