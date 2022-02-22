"""
Preliminary checks before the code is run.

Ensures that the correct folders exists that contain the data and that config exists etc.
"""
import os

from definitions import DATASET_PATH
from utility.file_util import create_dir


# Data for 10 subjects should exist - creates folders for each subject if none yet exist.
def check_data_folders():
    valid_subjects = list(range(9))

    for subject in valid_subjects:
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}'), recursive=True)
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'training'))
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'online_test'))
        create_dir(os.path.join(DATASET_PATH, f'subject_{subject}', 'dwell_tuning'))