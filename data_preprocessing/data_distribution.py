import collections
import random
import pandas as pd
from data_preprocessing.trigger_points import is_triggered
from classes import Window
from classes import Dataset
import numpy as np


def aggregate_data(device_data_pd: pd.DataFrame, freq_size: int, tp_table: pd.DataFrame, sample_rate: int = 1200) -> [
    pd.DataFrame]:
    list_of_datawindows = []

    for i in range(0, device_data_pd.shape[0], freq_size):
        # the label for the window is attached first. we base being 'triggered' whether the middle frequency is
        # recorded during the triggered timewindow.
        window = Window.Window()

        window.label = is_triggered(i + freq_size / 2, tp_table, sample_rate)
        window.data = device_data_pd.iloc[i:i + freq_size]

        list_of_datawindows.append(window)

    # return all but the last window, because it is not complete
    return list_of_datawindows[:-1]


# finds the start of trigger point and converts it to frequency and takes the window_size (in seconds) and cuts each
# side into a datawindow.
# this is used to find peaks locally in EMG data.
def cut_windows(tp_table: pd.DataFrame, tt_column: str, data: pd.DataFrame,
               dataset: Dataset, window_size: float = 2.) -> ([Window], Dataset):
    list_of_trigger_windows = []
    indices_to_delete = []

    for i, row in tp_table.iterrows():
        start = int(row[tt_column].total_seconds() * dataset.sample_rate - window_size * dataset.sample_rate)
        end = int(row[tt_column].total_seconds() * dataset.sample_rate + window_size * dataset.sample_rate)
        window = Window.Window()
        window.data = dataset.data_device1.iloc[start:end]
        window.label = 1  # indicates EMG
        window.timestamp = row

        window.filtered_data = data.iloc[start:end]
        window.filtered_data = window.filtered_data.reset_index(drop=True)
        indices_to_delete.append([start, end])
        list_of_trigger_windows.append(window)

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        data = data.drop(data.index[indices[0]:indices[1]])

    return list_of_trigger_windows, data, dataset


def slice_and_label_idle_windows(data: pd.DataFrame, filtered_data: pd.DataFrame, window_size: int = 2, freq: int = 1200) -> [Window]:
    list_of_windows = []
    window_sz = window_size * freq
    i = 0
    while i < len(data) and i + window_sz < len(data):
        cutout = abs(data.index[i] - data.index[i + window_sz]) == window_sz
        if cutout:
            window = Window.Window()
            window.data = data.iloc[i:i + window_sz]
            window.label = 0  # indicates no EMG peak / no MRCP should be present
            window.filtered_data = filtered_data.iloc[i:i + window_sz]
            window.filtered_data = window.filtered_data.reset_index(drop=True)
            list_of_windows.append(window)
            i += window_sz
        else:
            i += 1

    return list_of_windows


def data_distribution(labelled_data_lst: [Window]) -> {}:
    triggered = 0

    for window in labelled_data_lst:
        if window.label == 1:
            triggered += 1
    #  counter = collections.Counter(features)

    idle = len(labelled_data_lst) - triggered
    return {
        'triggered': triggered,
        'idle': idle,
        'expected_triggered_percent': int(triggered / (triggered + idle) * 100)
    }


def create_uniform_distribution(data_list: [Window]) -> [Window]:
    # returns the dataset with equal amount of samples, chosen by the least represented feature.
    features = []

    for window in data_list:
        features.append(window.label)

    counter = collections.Counter(features)
    least_represented_feature = 99999999999
    feat_counter = {}
    for i in counter.keys():
        feat_counter[i] = 0
        if counter[i] < least_represented_feature:
            least_represented_feature = counter[i]

    random.shuffle(data_list)

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
