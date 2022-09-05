import glob
import pandas as pd
import scipy.io
import os
from classes.Dataset import Dataset
from definitions import DATASET_PATH
from utility.logger import get_logger
import matplotlib.pyplot as plt


# Takes care of loading in the dataset into our Dataset class
def init(data, config, open_l=None, filename=None):
    dataset = Dataset()

    dataset.data = pd.DataFrame(data=data['data_device1'][:, :len(config.EEG_CHANNELS)],
                                columns=config.EEG_CHANNELS)

    if open_l:
        dataset.data[config.EMG_CHANNEL] = data['data_device1'][:, 12]
    else:
        dataset.data[config.EMG_CHANNEL] = data['data_device1'][:, 12]

    dataset.sample_rate = 1200
    dataset.filename = filename
    if config.rest_classification:
        dataset.label = 1
    else:
        dataset.label = open_l

    dataset.data[config.EOG_CHANNEL] = data['data_device1'][:, 9]

    return dataset


def create_dataset(path: str, config):
    train_data = []

    if config.transfer_learning and isinstance(path, list):
        for t_path in path:
            names, data, filenames = read_data(t_path)
            assert len(data) == len(names)
            for dataset, label, filename in zip(data, names, filenames):
                train_data.append(init(dataset, config, label, filename))
        return train_data

    names, data, filenames = read_data(path)

    if len(data) == 0:
        get_logger().error(f'No files found in {path}')
    elif len(data) == 1:
        # Dwell data
        return init(data[0], config, filename=filenames)
    elif len(data) == 2:
        online_test_data = []
        for dataset, filename in zip(data, filenames):
            online_test_data.append(init(dataset, config, filename=filename))
        return online_test_data
    else:
        train_data = []
        assert len(data) == len(names)
        for dataset, label, filename in zip(data, names, filenames):
            train_data.append(init(dataset, config, label, filename))
        return train_data


def read_data(path: str):
    names = []
    data = []
    filenames = []
    for file in glob.glob(path, recursive=True):
        if ".mat" in file:
            data.append(scipy.io.loadmat(file))
            filenames.append(file.lower().split('\\')[-3] + ' ' + file.lower().split('\\')[-1])
            if 'close' in file:
                names.append(0)
            elif 'open' in file:
                names.append(1)

    return names, data, filenames


def get_dataset_paths(subject_id: int, config):
    assert config.transfer_learning is not None
    if config.transfer_learning:
        training_p = []
        for sub_temp in range(10):
            if sub_temp == subject_id:
                continue
            training_p.append(os.path.join(DATASET_PATH, f'subject_{sub_temp}', 'training/*'))
    else:
        training_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'training/*')

    online_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'online_test/*')
    dwell_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'dwell_tuning/*')

    return training_p, online_p, dwell_p


def format_input(arg: str):
    arg = arg.lower()
    if arg == 'y':
        return True
    elif arg == 'n':
        return False


def print_module_options():
    print("1. A combination of features extracted from different deep learning algorithms will improve the classification.")
    print("2. The electrode layout will allow for differentiation between rest and movement in the EEG signals.")
    print("3. A model can be trained to predict a new subject without any calibration.")
    print("4. Calibration will improve the accuracy of the model when predicting on a new subject.")
    print("5. Deep feature extraction will generalize better to cross-session datasets compared to handcrafted features.")
    print("6. A high number of recorded movements from each subject will help improve classification of our models.")
    print("7. Analysis (SD, RSD, mean) of the data recorded from each session of each subject")
