import sys
from argparse import ArgumentParser, Namespace
import logging

from torch.utils.data import Dataset

from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

from typing import Dict, Type

from transformer_wrappers.utils.scripting import init_training_environment
from transformer_wrappers.data import corpus_mapping
from transformer_wrappers.wrappers import CausalLMWrapper, causal_lm_mapping


def main(args: Namespace):
    # Init environment
    configs: Dict = init_training_environment(args.config_file_path)
    # Start logging info
    logging.info("Script started and configuration file loaded")
    # Build model
    model_type: Type[CausalLMWrapper] = causal_lm_mapping[configs['model'].pop('dtype')]
    model: CausalLMWrapper = model_type(**configs['model'])
    logging.info("Model built")
    # Create data set splits
    corpus_dtype: Type[Dataset] = corpus_mapping[configs['data']['corpus']]
    data_splits: Dict[str, Dataset] = {
        split: corpus_dtype(
            split, model.tokenizer, **configs['data'].get('params', dict())
        ) for split in configs['data']['splits']
    }
    logging.info("Data set splits loaded")
    # Create callbacks
    callbacks = {
        'early_stopping': pl_callbacks.EarlyStopping(
            **configs.get('callbacks', dict()).get('early_stopping', dict())
        ),
        'model_checkpoint': pl_callbacks.ModelCheckpoint(
            **configs.get('callbacks', dict()).get('model_checkpoint', dict())
        ),
        'learning_rate_monitor': pl_callbacks.LearningRateMonitor()
    }
    logging.info("Callbacks instantiated")
    # Create loggers
    loggers = [
        pl_loggers.TensorBoardLogger(configs['current_experiment_dir_path']),
        pl_loggers.CSVLogger(configs['current_experiment_dir_path'])
    ]
    logging.info("Loggers instantiated")
    # Fit and evaluate model
    model.set_fine_tuning_params(**configs['hyperparameters'])
    model.fit_eval(
        data_splits, dir_path=configs['current_experiment_dir_path'], callbacks=callbacks, loggers=loggers
    )
    # Close script info
    logging.info("Script executed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='fit_eval_script',
        description='Script to fine-tune and evaluate a causal transformer LM wrapper'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the training configurations."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))

