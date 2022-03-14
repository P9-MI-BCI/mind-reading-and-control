from data_preprocessing.emg_processing import onset_detection
from data_preprocessing.init_dataset import init
import glob
import pandas as pd
import scipy.io
import os
from classes.Dataset import Dataset
from definitions import DATASET_PATH
from utility.logger import get_logger


def create_dataset_from_config(config, label_config):
    data = []
    names = []
    filenames = []
    for k in label_config:
        path = os.path.join(DATASET_PATH, k, '**/*.mat')
        for file in glob.glob(path, recursive=True):
            for trial in label_config[k]['emg_outliers']:
                if file.lower().endswith(trial.lower()):
                    data.append(scipy.io.loadmat(file))
                    filenames.append(file.lower().split('\\')[-3]+' '+file.lower().split('\\')[-1])
                    if 'close' in file:
                        names.append(0)
                    elif 'open' in file:
                        names.append(1)
                    else:
                        names.append(None)

    if len(data) == 0:
        get_logger().error(f'No files found in {path}')
    else:
        train_data = []
        assert len(data) == len(names)
        for dataset, label, filename in zip(data, names, filenames):
            train_data.append(init(dataset, config, label, filename))
        return train_data


def outlier_test(config, label_config):

    data = create_dataset_from_config(config, label_config)
    try:
        for dataset in data:
            onset_detection(dataset, config)
    except TypeError as te:
        get_logger().error("Dataset for EMG outlier detection is probably Nonetype, fix path/config file")



