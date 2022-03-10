"""
Preliminary checks before the code is run.

Ensures that the correct folders exists that contain the data and that config exists etc.
"""
import os
import json
import numpy as np

from classes.Dict import AttrDict
from definitions import ROOT_PATH, DATASET_PATH
from utility.file_util import create_dir, file_exist
from utility.logger import get_logger


# Makes sure the default config and online label config files exist.
def check_config_files():
    default_config_path = os.path.join(ROOT_PATH, 'json_configs/default.json')
    label_config_path = os.path.join(ROOT_PATH, 'json_configs/file_management.json')

    try:
        with open(default_config_path, 'r') as default_c, open(label_config_path) as label_c:
            return AttrDict(json.load(default_c)), AttrDict(json.load(label_c))
    except FileNotFoundError as e:
        get_logger().error(f'Config file missing at: {e.strerror}')


# Data for 10 subjects should exist - creates folders for each subject if none yet exist.
def check_data_folders():
    valid_subjects = list(range(9))

    for subject in valid_subjects:
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}'), recursive=True)
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'training'))
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'online_test'))
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'dwell_tuning'))


# Checks if the online_test data label files specified in the file_management.json config files exist.
def check_for_label_files(label_config):
    for k, v in label_config.items():
        path = os.path.join(DATASET_PATH, k, 'online_test')
        try:
            if not np.all(list(map(os.path.exists, [os.path.join(path, i) for i in v]))):
                raise FileNotFoundError(f'Label file missing at {k}')
        except FileNotFoundError as e:
            get_logger().error(e)
