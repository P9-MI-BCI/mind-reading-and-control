import collections
import random
import pandas as pd
import sys

from data_preprocessing.eog_detection import blink_detection
from data_preprocessing.trigger_points import is_triggered
from classes.Window import Window
from classes import Dataset
import numpy as np


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


def cut_mrcp_windows_rest_movement_phase(tp_table: pd.DataFrame, tt_column: str, filtered_data: pd.DataFrame, dataset: Dataset,
                     window_size: int) -> ([Window], Dataset):
    list_of_windows = []
    indices_to_delete = []

    window_sz = window_size * dataset.sample_rate

    for i, row in tp_table.iterrows():
        center = int(row[tt_column].total_seconds() * dataset.sample_rate)

        # [-1, 1]
        window0 = Window()
        window0.data = dataset.data_device1.iloc[center - window_sz: center + window_sz]
        window0.label = 1 # Movement phase
        window0.timestamp = row
        window0.frequency_range = [center - window_sz, center + window_sz]
        window0.filtered_data = filtered_data.iloc[center - window_sz: center + window_sz]
        window0.filtered_data = window0.filtered_data.reset_index(drop=True)
        list_of_windows.append(window0)

        #  indices_to_delete.append([center - window_sz, center + window_sz])

        # [-4, -2]
        window1 = Window()
        window1.data = dataset.data_device1.iloc[center - window_sz * 4: center - window_sz * 2]
        window1.label = 0  # Rest phase
        window1.timestamp = row
        window1.frequency_range = [center - window_sz * 4, center - window_sz * 2]
        window1.filtered_data = filtered_data.iloc[center - window_sz * 4: center - window_sz * 2]
        window1.filtered_data = window1.filtered_data.reset_index(drop=True)
        list_of_windows.append(window1)

        # indices_to_delete.append([center - window_sz * 4, center + window_sz * 2])

        # indices_to_delete.reverse()

        # for indices in indices_to_delete:
        #     dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        #     filtered_data = filtered_data.drop(filtered_data.index[indices[0]:indices[1]])

    return list_of_windows # , filtered_data, dataset


def cut_mrcp_windows_for_online(tp_table: pd.DataFrame, tt_column: str, filtered_data: pd.DataFrame, dataset: Dataset,
                                window_size: int) -> (
        [Window], Dataset):
    list_of_trigger_windows = []
    indices_to_delete = []

    window_sz = window_size * dataset.sample_rate
    for i, row in tp_table.iterrows():
        start = int(row[tt_column].total_seconds() * dataset.sample_rate - window_sz)
        end = int(row[tt_column].total_seconds() * dataset.sample_rate + window_sz)

        window0 = Window()
        window0.data = dataset.data_device1.iloc[start - window_sz: end - window_sz]
        window0.label = 1
        window0.timestamp = row
        window0.frequency_range = [start - window_sz, end - window_sz]
        window0.filtered_data = filtered_data.iloc[start - window_sz: end - window_sz]
        window0.filtered_data = window0.filtered_data.reset_index(drop=True)
        list_of_trigger_windows.append(window0)

        window1 = Window()
        window1.data = dataset.data_device1.iloc[start - window_sz // 2: end - window_sz // 2]
        window1.label = 1  # indicates EMG
        window1.timestamp = row
        window1.frequency_range = [start, end]
        window1.filtered_data = filtered_data.iloc[start - window_sz // 2: end - window_sz // 2]
        window1.filtered_data = window1.filtered_data.reset_index(drop=True)
        list_of_trigger_windows.append(window1)

        # window2 = Window()
        # window2.data = dataset.data_device1.iloc[start + window_sz: end + window_sz]
        # window2.label = 1
        # window2.timestamp = row
        # window2.frequency_range = [start + window_sz, end + window_sz]
        # list_of_trigger_windows.append(window2)

        indices_to_delete.append([start - window_sz, end - window_sz // 2])

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        filtered_data = filtered_data.drop(filtered_data.index[indices[0]:indices[1]])

    return list_of_trigger_windows, filtered_data, dataset


def cut_and_label_idle_windows(data: pd.DataFrame, filtered_data: pd.DataFrame,
                               window_size: int, freq: int) -> [Window]:
    list_of_windows = []
    window_sz = window_size * freq * 2
    i = 0

    while i < len(data) and i + window_sz < len(data):
        cutout = abs(data.index[i] - data.index[i + window_sz]) == window_sz
        if cutout:
            window = Window()
            window.data = data.iloc[i:i + window_sz]
            window.label = 0  # indicates no MRCP should be present
            window.frequency_range = [data.index[i], data.index[i] + window_sz]

            window.filtered_data = filtered_data.iloc[i:i + window_sz]
            window.filtered_data = window.filtered_data.reset_index(drop=True)
            list_of_windows.append(window)
            i += window_sz
        else:
            i += 1

    return list_of_windows


def slice_and_label_idle_windows_for_online(data: pd.DataFrame, window_size: int, freq: int) -> [Window]:
    list_of_windows = []
    window_sz = window_size * freq * 2
    i = 0
    while i < len(data) and i + window_sz < len(data):
        cutout = abs(data.index[i] - data.index[i + window_sz]) == window_sz
        if cutout:
            window = Window()
            window.data = data.iloc[i:i + window_sz]
            window.label = 0  # indicates no EMG peak / no MRCP should be present
            window.frequency_range = [data.index[i], data.index[i] + window_sz]
            list_of_windows.append(window)
            i += window_sz
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
