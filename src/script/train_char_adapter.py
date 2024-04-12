import sys
from argparse import ArgumentParser, Namespace
import logging
from datetime import datetime

from typing import Dict

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

from transformer_wrappers.wrappers import TransformerWrapper
from transformer_wrappers.wrappers.char import *

from transformer_wrappers.utils import init_training_environment
from transformer_wrappers.data import TokeNNDataset


def main(args: Namespace):
    # Init environment
    configs: Dict = init_training_environment(args.config_file_path)
    # Start logging info
    logging.info("Script started and configuration file loaded")
    # Load target (L)LM embedding weights and tokeniser
    transformer = TransformerWrapper.from_pretrained(**configs['lm'])
    embeddings = transformer.embedding
    tokenizer = transformer.tokenizer
    del transformer  # TODO find better solution
    # Create NN
    tokenn: TokeNN = TokeNN.from_pretrained(**configs['tokenn'])
    tokenn.configure_metrics()
    # Start Logging info
    logging.info("Neural network loaded")
    # Create data set splits
    data_splits: Dict[str, TokeNNDataset] = {
        split: TokeNNDataset(
            split,
            embeddings,
            tokenizer,
            tokenn.char_tokenizer,
            **configs['data'].get('params', dict())
        )
        for split in configs['data']['splits']
    }
    logging.info("Data set splits loaded")
    # Create data loaders
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(
            data,
            collate_fn=data.collate,
            shuffle=split == 'train',
            **configs['data']['loader'][split]
        )
        for split, data in data_splits.items()
    }
    logging.info("Data loaders instantiated")
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
    # Instantiate Trainer object with the callbacks
    trainer = L.Trainer(
        default_root_dir=configs['current_experiment_dir_path'],
        **configs.get('trainer', dict()),
        callbacks=list(callbacks.values()),
        logger=loggers
    )
    logging.info("Trainer instantiated")
    # Train neural network
    start_time = datetime.now()
    logging.info("Training started")
    trainer.fit(tokenn, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])
    stop_time = datetime.now()
    logging.info(f"Training completed (elapsed time: {stop_time - start_time})")
    # Load torch checkpoint
    checkpoint = torch.load(callbacks['checkpoint_callback'].best_model_path)
    tokenn.load_state_dict(checkpoint['state_dict'])
    logging.info(f"Best checkpoint restored from {callbacks['checkpoint_callback'].best_model_path}")
    # Test neural network
    start_time = datetime.now()
    logging.info("Validation started")
    trainer.validate(tokenn, dataloaders=data_loaders['validation'])
    stop_time = datetime.now()
    logging.info(f"Validation completed (elapsed time: {stop_time - start_time})")
    start_time = datetime.now()
    logging.info("Testing started")
    trainer.test(tokenn, dataloaders=data_loaders['test'])
    stop_time = datetime.now()
    logging.info(f"Testing completed (elapsed time: {stop_time - start_time})")
    # Close script info
    logging.info("Script executed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='train_char_adapter',
        description='Script to train a character level adapter starting from a pre-trained transformer'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the training configurations."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
