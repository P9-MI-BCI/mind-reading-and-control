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

# Finds the start of TriggerPoints and converts it to frequency and takes the window_size (in seconds) and cuts each
# side into a datawindow. This is used to find peaks locally in EMG data.
def cut_mrcp_windows(tp_table: pd.DataFrame, tt_column: str, filtered_data: pd.DataFrame, dataset: Dataset,
                     window_size: int) -> ([Window], Dataset):
    list_of_mrcp_windows = []
    indexes_to_delete = []
    window_sz = dataset.sample_rate * window_size
    for i, row in tp_table.iterrows():
        start = int(row[tt_column].total_seconds() * dataset.sample_rate - window_sz * 1.5)
        end = int(row[tt_column].total_seconds() * dataset.sample_rate + window_sz // 2)
        window = Window()
        window.data = dataset.data_device1.iloc[start:end]
        window.label = 1  # indicates MRCP within window
        window.timestamp = row
        window.frequency_range = [start, end]

        window.filtered_data = filtered_data.iloc[start:end]
        window.filtered_data = window.filtered_data.reset_index(drop=True)
        indexes_to_delete.append([start, end])
        list_of_mrcp_windows.append(window)

    indexes_to_delete.reverse()

    # Cuts out the indexes of the detected MRCP windows from the data
    for indexes in indexes_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indexes[0]:indexes[1]])
        filtered_data = filtered_data.drop(filtered_data.index[indexes[0]:indexes[1]])

    return list_of_mrcp_windows, filtered_data, dataset


def cut_mrcp_windows_rest_movement_phase(tp_table: pd.DataFrame, tt_column: str, filtered_data: pd.DataFrame,
                                         dataset: Dataset,
                                         window_size: int, sub_windows: bool = False, multiple_windows: bool = True,
                                         perfect_centering: bool = False) -> ([Window], Dataset):
    list_of_windows = []
    indices_to_delete = []

    window_sz = int((window_size * dataset.sample_rate) / 2)

    # sub windows are half a second
    sub_window_sz = int(window_sz / 2)
    # step size is 50% overlap
    step_sz = int(sub_window_sz / 2)

    # weights
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # ids
    id = 0
    for i, row in tp_table.iterrows():
        center = int(row[tt_column].total_seconds() * dataset.sample_rate)

        window0 = Window()
        window0.filtered_data = filtered_data.iloc[center - window_sz: center + window_sz]

        if perfect_centering:
            adjustments = []
            for column in window0.filtered_data.columns:
                if column == 12 or column == 9:
                    continue
                distance = (window0.filtered_data[column].idxmin() - center) * weights[column]
                adjustments.append(distance)
            negative_counter = 0
            for d in adjustments:
                if d <= 0:
                    negative_counter += 1
            if negative_counter > len(adjustments) / 2:
                adjustments = [min(x, 0) for x in adjustments]
            else:
                adjustments = [max(x, 0) for x in adjustments]

            adjustments = [i for i in adjustments if i != 0]
            adjustments.sort()
            temp_mean = int(median(adjustments))
            if temp_mean > 0:
                adjustments = [i for i in adjustments if temp_mean * 1.5 > i]
            else:
                adjustments = [i for i in adjustments if temp_mean * 1.5 < i]

            if adjustments:
                center = int(center + median(adjustments))

        # [-1, 1]
        w0_id = f'mrcp:{id}'
        window0.data = dataset.data_device1.iloc[center - window_sz: center + window_sz]
        window0.label = 1  # Movement phase
        window0.timestamp = row
        window0.frequency_range = [center - window_sz, center + window_sz]
        window0.filtered_data = filtered_data.iloc[center - window_sz: center + window_sz]
        window0.filtered_data = window0.filtered_data.reset_index(drop=True)
        window0.num_id = w0_id
        window0.is_sub_window = False

        list_of_windows.append(window0)
        id += 1

        indices_to_delete.append([center - window_sz, center + window_sz])

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        filtered_data = filtered_data.drop(filtered_data.index[indices[0]:indices[1]])

    return list_of_windows


def cut_mrcp_windows_calibration(tp_table: pd.DataFrame, tt_column: str, filtered_data: pd.DataFrame, dataset: Dataset,
                                 window_size: int, weights=None, multiple_windows: bool = True,
                                 perfect_centering: bool = False, ) -> ([Window], Dataset):
    list_of_windows = []
    indices_to_delete = []
    window_sz = int((window_size * dataset.sample_rate) / 2)

    # weights
    if weights is None:
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # ids
    id = 0
    for i, row in tp_table.iterrows():
        center = int(row[tt_column].total_seconds() * dataset.sample_rate)

        window0 = Window()
        window0.filtered_data = filtered_data.iloc[center - window_sz: center + window_sz]

        if perfect_centering:
            adjustments = []
            for column in window0.filtered_data.columns:
                if column == 12 or column == 9:
                    continue
                distance = (window0.filtered_data[column].idxmin() - center)
                adjustments.append(distance)
            negative_votes = 0
            positive_votes = 0

            for d in range(0, len(adjustments)):
                if adjustments[d] <= 0:
                    negative_votes += weights[d]
                else:
                    positive_votes += weights[d]

            if negative_votes > positive_votes:
                adjustments = [min(x, 0) for x in adjustments]
            else:
                adjustments = [max(x, 0) for x in adjustments]

            adjustments = [i for i in adjustments if i != 0]
            adjustments.sort()
            temp_mean = int(median(adjustments))
            if temp_mean > 0:
                adjustments = [i for i in adjustments if temp_mean * 1.5 > i]
            else:
                adjustments = [i for i in adjustments if temp_mean * 1.5 < i]

            if adjustments:
                center = int(center + median(adjustments))

        # [-1, 1]
        w0_id = f'mrcp:{id}'
        window0.data = dataset.data_device1.iloc[center - window_sz: center + window_sz]
        window0.label = 1  # Movement phase
        window0.timestamp = row
        window0.frequency_range = [center - window_sz, center + window_sz]
        window0.filtered_data = filtered_data.iloc[center - window_sz: center + window_sz]
        window0.filtered_data = window0.filtered_data.reset_index(drop=True)
        window0.num_id = w0_id
        window0.is_sub_window = False

        list_of_windows.append(window0)
        id += 1

        indices_to_delete.append([center - window_sz, center + window_sz])

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        filtered_data = filtered_data.drop(filtered_data.index[indices[0]:indices[1]])

    return list_of_windows


def cut_and_label_idle_windows(data: pd.DataFrame, filtered_data: pd.DataFrame,
                               window_size: int, freq: int) -> [Window]:
    list_of_windows = []
    window_sz = int(window_size * freq)
    i = 0

    id = 0
    while i < len(data) and i + window_sz < len(data):
        cutout = abs(data.index[i] - data.index[i + window_sz]) == window_sz
        if cutout:
            rest_id = f'rest:{id}'
            window = Window()
            window.data = data.iloc[i:i + window_sz]
            window.label = 0  # indicates no MRCP should be present
            window.frequency_range = [data.index[i], data.index[i] + window_sz]
            window.id = rest_id
            window.is_sub_window = False
            window.filtered_data = filtered_data.iloc[i:i + window_sz]
            window.filtered_data = window.filtered_data.reset_index(drop=True)
            list_of_windows.append(window)
            i += window_sz
            id += 1
        else:
            i += 1

    return list_of_windows


def data_distribution(labelled_data_lst: [Window]) -> {}:
    labeled_window = 0

    for window in labelled_data_lst:
        if window.label == 1:
            labeled_window += 1
    #  counter = collections.Counter(features)

    idle = len(labelled_data_lst) - labeled_window
    return {
        'labeled': labeled_window,
        'idle': idle,
        'expected_labeled_percent': int(labeled_window / (labeled_window + idle) * 100)
    }


def create_uniform_distribution(data_list: [Window]) -> [Window]:
    # Returns the dataset with equal amount of samples, chosen by the least represented feature.
    features = []

    for window in data_list:
        features.append(window.label)

    counter = collections.Counter(features)  # find how many of each feature is present
    least_represented_feature = sys.maxsize
    feat_counter = {}

    # find the value of how many exist of the least represented feature
    for i in counter.keys():
        feat_counter[i] = 0
        if counter[i] < least_represented_feature:
            least_represented_feature = counter[i]

    random.shuffle(data_list)

    # takes random samples of each feature until the amount is equal to the least feature amount
    uniform_data_list = []
    for window in data_list:
        if feat_counter[window.label] < least_represented_feature:
            uniform_data_list.append(window)
            feat_counter[window.label] += 1

    return uniform_data_list


def z_score_normalization(window: pd.DataFrame) -> pd.DataFrame:
    return (window - window.mean()) / window.std()


def max_absolute_scaling(window: pd.DataFrame) -> pd.DataFrame:
    return window / window.abs().max()


def min_max_scaling(window: pd.DataFrame) -> pd.DataFrame:
    return (window - window.min()) / (window.max() - window.min())


def blinks_in_list(windows: [Window]) -> int:
    counter = 0
    for window in windows:
        if window.blink == 1:
            counter += 1

    return counter


def data_preparation(datasets, config):
    X = []
    Y = []

    if config.rest_classification:
        for dataset in datasets:
            step_i = 0

            while step_i < len(dataset.data) - int(config.window_size * dataset.sample_rate):
                is_in_cluster = []
                for cluster in dataset.onsets_index:
                    if cluster[0] - dataset.sample_rate / 2 < step_i < cluster[0]:
                        is_in_cluster.append(True)
                    elif not cluster[0] - dataset.sample_rate / 2 < step_i < cluster[2]:
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
            for cluster in dataset.onsets_index:
                if cluster[0] - config.window_size * dataset.sample_rate < 0:
                    continue
                X.append(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                         cluster[0] - int(config.window_size * dataset.sample_rate):
                         cluster[0] + int(config.window_size * dataset.sample_rate)].to_numpy())
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
                for cluster in dataset.onsets_index:
                    # half a second before and after the cluster onset
                    if cluster[0] - dataset.sample_rate / 2 < step_i < cluster[0] + dataset.sample_rate / 2:
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
                skip_mark = False
                for cluster in dataset.onsets_index:
                    if cluster[0] - dataset.sample_rate * 1.5 < step_i < cluster[0]:
                        is_in_cluster.append(True)
                        break
                    elif not cluster[0] - dataset.sample_rate / 2 < step_i < cluster[2]:
                        is_in_cluster.append(False)
                    elif cluster[0] < step_i < cluster[2]:
                        skip_mark = True
                        break
                if skip_mark:
                    step_i += int(config.step_size * dataset.sample_rate)
                    continue
                elif any(is_in_cluster):
                    Y.append(1)

                    sliding_window = filter_module(config, config.DELTA_BAND, data_buffer, dataset.sample_rate)
                    X.append(sliding_window)
                # if current step is not within a movement cluster and there are more 1 than 0 labels,
                # add a new 0 label
                elif not any(is_in_cluster) and sum(Y) > len(Y) / 2:
                    Y.append(0)

                    sliding_window = filter_module(config, config.DELTA_BAND, data_buffer, dataset.sample_rate)
                    X.append(sliding_window)

                step_i += int(config.step_size * dataset.sample_rate)

            with open(os.path.join(OUTPUT_PATH, f'{temp_file_name}{dataset_num}'), 'wb') as f:
                pickle.dump(X, f)

            with open(os.path.join(OUTPUT_PATH, f'{temp_label_name}{dataset_num}'), 'wb') as f:
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
    for file in glob.glob(os.path.join(OUTPUT_PATH, '*'), recursive=True):
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