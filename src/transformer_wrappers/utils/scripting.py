import os
import logging
from shutil import copy2
from datetime import datetime

from itertools import product

import lightning as L
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

import yaml

from typing import Dict


def init_training_environment(config_file_path: str) -> Dict:
    # Get datetime string
    date_time: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Load configs
    with open(config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Paths and directories
    experiments_dir_path: str = configs['experiments_dir_path']
    if not os.path.exists(experiments_dir_path):
        os.mkdir(experiments_dir_path)
    experiment_series_dir_path: str = os.path.join(experiments_dir_path, configs['experiment_series'])
    if not os.path.exists(experiment_series_dir_path):
        os.mkdir(experiment_series_dir_path)
    current_experiment_dir_path: str = os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time}"
    )
    configs['current_experiment_dir_path'] = current_experiment_dir_path
    if not os.path.exists(current_experiment_dir_path):
        os.mkdir(current_experiment_dir_path)
    # Set random seed
    L.seed_everything(configs.get('random_seed'), workers=True)
    # Dump configs
    config_dump_file_path = os.path.join(current_experiment_dir_path, f'config.yml')
    copy2(config_file_path, config_dump_file_path)
    # Complete configurations setup
    if configs['model'].get('quantization_configs') is not None:
        configs['model']['quantization_configs'] = BitsAndBytesConfig(**configs['model']['quantization_configs'])
    if configs['model'].get('lora_configs') is not None:
        configs['model']['lora_configs'] = LoraConfig(**configs['model']['lora_configs'])
    # Init logging
    if configs.get('log_file', False):
        log_file_path = os.path.join(current_experiment_dir_path, f'training.log')
    else:
        log_file_path = None
    logging.basicConfig(filename=log_file_path, level=configs.get('log_level', 'INFO'))

    return configs


def init_evaluation_environment(config_file_path: str) -> Dict:
    # Load configs
    with open(config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Paths and directories
    experiments_dir_path: str = configs['experiments_dir_path']
    if not os.path.exists(experiments_dir_path):
        os.mkdir(experiments_dir_path)
    experiment_series_dir_path: str = os.path.join(experiments_dir_path, configs['experiment_series'])
    if not os.path.exists(experiment_series_dir_path):
        os.mkdir(experiment_series_dir_path)
    current_experiment_dir_path: str = os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}"
    )
    configs['current_experiment_dir_path'] = current_experiment_dir_path
    if not os.path.exists(current_experiment_dir_path):
        os.mkdir(current_experiment_dir_path)
    # Set random seed
    L.seed_everything(configs.get('random_seed'), workers=True)
    # Dump configs
    config_dump_file_path = os.path.join(current_experiment_dir_path, f'config.yml')
    copy2(config_file_path, config_dump_file_path)
    # Complete configurations setup
    configs['model']['model_kwargs']['torch_dtype'] = torch.bfloat16  # TODO find better solution
    if configs['model'].get('quantization_configs') is not None:
        configs['model']['quantization_configs'] = BitsAndBytesConfig(**configs['model']['quantization_configs'])
    if configs['model'].get('lora_configs') is not None:
        configs['model']['lora_configs'] = LoraConfig(**configs['model']['lora_configs'])
    # Prepare evaluation configs
    params = configs['exp'].pop('params', dict())
    params = [{k: v for k, v in zip(params.keys(), param_values)} for param_values in product(*list(params.values()))]
    param_groups = configs['exp'].pop('param_groups', list() if len(params) > 0 else [dict()])
    configs['exp']['params'] = params + param_groups
    # Init logging
    if configs.get('log_file', False):
        log_file_path = os.path.join(current_experiment_dir_path, f'training.log')
    else:
        log_file_path = None
    logging.basicConfig(filename=log_file_path, level=configs.get('log_level', 'INFO'))

    return configs
