import glob
import pandas as pd
import scipy.io
import os
from classes.Dataset import Dataset
from definitions import DATASET_PATH
from utility.logger import get_logger


# Takes care of loading in the dataset into our Dataset class
def init(data, config, open_l=None):
    dataset = Dataset()

    dataset.data = pd.DataFrame(data=data['data_device1'][:, :len(config.EEG_CHANNELS)],
                                columns=config.EEG_CHANNELS)

    # todo update whenever we know the emg channels
    if open_l:
        dataset.data[config.EMG_CHANNEL] = data['data_device1'][:, 12]
    else:
        dataset.data[config.EMG_CHANNEL] = data['data_device1'][:, 12]
    dataset.sample_rate = 1200
    dataset.label = open_l

    return dataset


def create_dataset(path: str, config):
    data = []
    names = []
    for file in glob.glob(path, recursive=True):
        data.append(scipy.io.loadmat(file))
        if 'close' in file:
            names.append(0)
        elif 'open' in file:
            names.append(1)

    if len(data) == 0:
        get_logger().error(f'No files found in {path}')
    elif len(data) == 1:
        return init(data[0], config)
    else:
        train_data = []
        assert len(data) == len(names)
        for dataset, label in zip(data, names):
            train_data.append(init(dataset, config, label))
        return train_data


def get_dataset_paths(subject_id: int):
    training_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'training/*')
    online_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'online_test/*')
    dwell_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'dwell_tuning/*')

    return training_p, online_p, dwell_p
