import os
import sys
import logging
from shutil import copy2
from datetime import datetime

import torch
import lightning as L

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
    # Init logging
    if configs.get('log_file', False):
        log_file_path = os.path.join(current_experiment_dir_path, f'training.log')
    else:
        log_file_path = None
    logging.basicConfig(filename=log_file_path, level=configs.get('log_level', 'INFO'))

    return configs
