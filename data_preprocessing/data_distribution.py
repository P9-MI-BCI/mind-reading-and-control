import collections
import glob
import random
import pandas as pd
import numpy as np
import sys
import os
import json

from statistics import mean, stdev, median

from tqdm import tqdm

from classes import Window, Dataset
from sklearn.preprocessing import StandardScaler

from data_preprocessing.filters import butter_filter
from utility.file_util import file_exist
from definitions import DATASET_PATH, OUTPUT_PATH
import pickle
import os


def z_score_normalization(window: pd.DataFrame) -> pd.DataFrame:
    return (window - window.mean()) / window.std()


def max_absolute_scaling(window: pd.DataFrame) -> pd.DataFrame:
    return window / window.abs().max()


def min_max_scaling(window: pd.DataFrame) -> pd.DataFrame:
    return (window - window.min()) / (window.max() - window.min())


def data_preparation(datasets, config):
    X = []
    Y = []

    if config.rest_classification:
        for dataset in datasets:
            step_i = 0

            while step_i < len(dataset.data) - int(config.window_size * dataset.sample_rate):
                is_in_cluster = []
                for cluster in dataset.clusters:
                    if cluster.start - dataset.sample_rate / 2 < step_i < cluster.start:
                        is_in_cluster.append(True)
                    elif not cluster.start - dataset.sample_rate / 2 < step_i < cluster.strat:
                        is_in_cluster.append(False)

                if any(is_in_cluster):
                    Y.append(1)

                    X.append(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                             step_i:
                             step_i + int(config.window_size * dataset.sample_rate)
                             ])
                # if current step is not within a movement cluster and there are more 1 than 0 labels,
                # add a new 0 label
                elif not any(is_in_cluster) and sum(Y) > len(Y) / 2:
                    Y.append(0)

                    X.append(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                             step_i:
                             step_i + int(config.window_size * dataset.sample_rate)
                             ])

                step_i += int(config.step_size * dataset.sample_rate)

            if len(X) > 500:
                break

    elif not config.rest_classification:
        for dataset in datasets:
            for cluster in dataset.clusters:
                if cluster.start - config.window_size * dataset.sample_rate < 0:
                    continue
                X.append(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                         cluster.start - int(config.window_size * dataset.sample_rate):
                         cluster.start + int(config.window_size * dataset.sample_rate)].to_numpy())
                Y.append(dataset.label)

    shuffler = np.random.permutation(len(X))
    X = np.array(X)[shuffler]
    Y = np.array(Y)[shuffler]

    return X, Y


def normalization(X):
    scaler = StandardScaler()

    flat_x = np.concatenate((X), axis=0)
    scaler.fit(flat_x)
    transformed_x = []
    for x in X:
        transformed_x.append(scaler.transform(x))

    return np.array(transformed_x), scaler


def online_data_labeling(datasets: [Dataset], config, scaler, subject_id: int):
    online_data_labels = get_online_data_labels(subject_id)
    X = []
    Y = []
    label_determiner = 0

    if config.rest_classification:
        for dataset in datasets:
            step_i = 0

            while step_i < len(dataset.data) - int(config.window_size * dataset.sample_rate):
                is_in_cluster = []
                for cluster in dataset.clusters:
                    # half a second before and after the cluster onset
                    if cluster.start - dataset.sample_rate / 2 < step_i < cluster.start + dataset.sample_rate / 2:
                        is_in_cluster.append(True)
                    else:
                        is_in_cluster.append(False)

                if any(is_in_cluster):
                    Y.append(1)

                    X.append(
                        scaler.transform(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                                         step_i:
                                         step_i + int(config.window_size * dataset.sample_rate)
                                         ]))
                # if current step is not within a movement cluster and there are more 1 than 0 labels,
                # add a new 0 label
                elif not any(is_in_cluster) and sum(Y) > len(Y) / 2:
                    Y.append(0)

                    X.append(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                             step_i:
                             step_i + int(config.window_size * dataset.sample_rate)
                             ])

                step_i += int(config.step_size * dataset.sample_rate)
            if len(X) > 1000:
                break

    if len(online_data_labels) == 1:
        if online_data_labels[0][1] == 1:
            for index, label in enumerate(online_data_labels[0][0]):
                Y[index] = label
        if online_data_labels[0][1] == 2:
            for i in range(len(online_data_labels[0][0])):
                Y[-i] = online_data_labels[0][0][-i]
    if len(online_data_labels) == 2:
        Y = online_data_labels[0][0].extend(online_data_labels[1][0])

    shuffler = np.random.permutation(len(X))
    X = np.array(X)[shuffler]
    Y = np.array(Y)[shuffler]

    return X, Y


def get_online_data_labels(subject_id: int):
    paths = [os.path.join(DATASET_PATH, f'subject_{subject_id}', 'online_test', 'online_01_labels.txt'),
             os.path.join(DATASET_PATH, f'subject_{subject_id}', 'online_test', 'online_02_labels.txt')]

    test_data_labels = []
    if file_exist(paths[0]):
        test_data_id = 1
        with open(paths[test_data_id - 1], encoding='utf-8') as f:
            labels = json.loads(f.read())
            test_data_labels.append([labels, test_data_id])

    if file_exist(paths[1]):
        test_data_id = 2
        with open(paths[test_data_id - 1], encoding='utf-8') as f:
            labels = json.loads(f.read())
            test_data_labels.append([labels, test_data_id])

    return test_data_labels


def data_preparation_with_filtering(datasets, config):
    dataset_num = 0
    temp_file_name = 'temp_data_'
    temp_label_name = 'temp_label_'
    if config.rest_classification:
        for dataset in tqdm(datasets):
            X = []
            Y = []
            data_buffer = pd.DataFrame(columns=config.EEG_CHANNELS)
            step_i = 0

            while step_i < len(dataset.data) - int(config.window_size * dataset.sample_rate):
                while len(data_buffer) < config.buffer_size * dataset.sample_rate:
                    data_buffer = pd.concat([
                        data_buffer,
                        dataset.data.iloc[step_i:
                                          step_i + int(dataset.sample_rate * config.step_size)]
                    ],
                        ignore_index=True)
                    step_i += int(config.step_size * dataset.sample_rate)

                data_buffer = pd.concat([data_buffer.iloc[
                                         int(dataset.sample_rate * config.step_size):],
                                         dataset.data.iloc[
                                         step_i:
                                         step_i + int(dataset.sample_rate * config.step_size)]
                                         ],
                                        ignore_index=True)

                is_in_cluster = []
                skip = False
                for cluster in dataset.clusters:
                    # 2 seconds before 1 sec after onset
                    if cluster.start - dataset.sample_rate * config.window_size < step_i < cluster.start:
                        is_in_cluster.append(True)
                        break
                    # leave half a second after movement ends to start labeling false
                    elif not cluster.start - dataset.sample_rate * config.window_size < step_i < cluster.end + dataset.sample_rate / 2:
                        is_in_cluster.append(False)
                    # if step is in the cluster just skip it.
                    elif cluster.start < step_i < cluster.end + dataset.sample_rate / 2:
                        skip = True
                        break
                if skip:
                    step_i += int(config.step_size * dataset.sample_rate)
                    continue
                elif any(is_in_cluster):
                    Y.append(1)

                    sliding_window = filter_module(config, config.DELTA_BAND, data_buffer, dataset.sample_rate)
                    X.append(sliding_window)
                elif not any(is_in_cluster) and sum(Y) > len(Y) / 2:
                    Y.append(0)

                    sliding_window = filter_module(config, config.DELTA_BAND, data_buffer, dataset.sample_rate)
                    X.append(sliding_window)

                step_i += int(config.step_size * dataset.sample_rate)
                
            with open(os.path.join(OUTPUT_PATH, 'data', f'{temp_file_name}{dataset_num}'), 'wb') as f:
                 pickle.dump(X, f)
          
            with open(os.path.join(OUTPUT_PATH, 'data',f'{temp_label_name}{dataset_num}'), 'wb') as f:
                 pickle.dump(Y, f)

            dataset_num += 1


def filter_module(config, filter_range, data_buffer, sample_rate):
    filtered_data = pd.DataFrame(columns=config.EEG_CHANNELS)

    for channel in config.EEG_CHANNELS:
        filtered_data[channel] = butter_filter(data=data_buffer[channel],
                                               order=config.EEG_ORDER,
                                               cutoff=filter_range,
                                               btype=config.EEG_BTYPE
                                               )

    sliding_window = np.array(filtered_data.iloc[-int(config.window_size*sample_rate):].reset_index(drop=True))
    return sliding_window


def load_data_from_temp():
    labels = []
    data = []
    path = os.path.join(OUTPUT_PATH, 'data')
    for file in glob.glob(os.path.join(path, '*'), recursive=True):
        if 'label' in file:
            with open(file, 'rb') as f:
                labels.extend(pickle.load(f))
        elif 'data' in file:
            with open(file, 'rb') as f:
                data.extend(pickle.load(f))

    return data, labels


def shuffle(X, Y):
    shuffler = np.random.permutation(len(X))
    X = np.array(X)[shuffler]
    Y = np.array(Y)[shuffler]
    return X, Y


def features_to_file(X, Y, scaler):
    with open(os.path.join(OUTPUT_PATH, 'features', 'features'), 'wb') as f:
        pickle.dump(X, f)

    with open(os.path.join(OUTPUT_PATH, 'features', 'labels'), 'wb') as f:
        pickle.dump(Y, f)

    with open(os.path.join(OUTPUT_PATH, 'scaler'), 'wb') as f:
        pickle.dump(scaler, f)


def load_features_from_file():
    labels = []
    data = []
    path = os.path.join(OUTPUT_PATH, 'features')
    for file in glob.glob(os.path.join(path, '*'), recursive=True):
        if 'labels' in file:
            with open(file, 'rb') as f:
                labels.extend(pickle.load(f))
        elif 'features' in file:
            with open(file, 'rb') as f:
                data.extend(pickle.load(f))

    return np.array(data), np.array(labels)


def load_scaler():
    with open(os.path.join(OUTPUT_PATH, 'scaler'), 'rb') as f:
        return pickle.load(f)
