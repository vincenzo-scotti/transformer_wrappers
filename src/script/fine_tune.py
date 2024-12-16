import sys
from argparse import ArgumentParser, Namespace
import logging
from datetime import datetime

from typing import Dict, Type

from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback, WandbCallback
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollator

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
    model: CausalLMWrapper = model_type.from_pretrained(**configs['model'])
    logging.info("Model built")
    data_splits = dict()
    for split, split_configs in configs['data'].items():
        data_splits[split] = corpus_mapping[split_configs['corpus']](
            split=split, tokenizer=model.tokenizer, **split_configs.get('params', dict())
        ) # TODO add support for hugging face built in data sets
    logging.info("Data set splits loaded")
    # Create callbacks
    callbacks = [
        EarlyStoppingCallback(**configs.get('callbacks', dict()).get('early_stopping', dict())),
        TensorBoardCallback(**configs.get('callbacks', dict()).get('tensorboard', dict())),
        WandbCallback()
    ]
    logging.info("Callbacks instantiated")
    # Create trainer
    training_args: TrainingArguments = TrainingArguments(
        output_dir=configs['current_experiment_dir_path'],
        **configs['hyperparameters']
    )
    logging.info("Training arguments prepared")
    data_collator: DataCollator = model.get_data_collator(**configs.get('collator', dict()))
    logging.info("Data collator instantiated")
    trainer: SFTTrainer = model.get_trainer(
        args=training_args,
        train_dataset=data_splits['train'],
        eval_dataset=data_splits['validation'],
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        callbacks=callbacks,
        **configs.get('trainer', dict())
    )
    logging.info("Trainer instantiated")
    # Sed model
    # Fit and evaluate model
    start_time = datetime.now()
    logging.info("Training started")
    trainer.train()
    stop_time = datetime.now()
    logging.info(f"Training completed (elapsed time: {stop_time - start_time})")
    # Save model
    model.save_pretrained(configs['current_model_dir_path'])
    logging.info(f"Model saved at `{configs['current_model_dir_path']}`")
    # Close script info
    logging.info("Script executed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='fine_tune_lm_script',
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
