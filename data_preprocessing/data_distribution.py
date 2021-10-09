import collections
import random
import pandas as pd
from data_preprocessing.trigger_points import is_triggered
from classes import Frame
from classes import Dataset
import numpy as np


def aggregate_data(device_data_pd: pd.DataFrame, freq_size: int, tp_table: pd.DataFrame, sample_rate: int = 1200) -> [
    pd.DataFrame]:
    list_of_dataframes = []

    for i in range(0, device_data_pd.shape[0], freq_size):
        # the label for the frame is attached first. we base being 'triggered' whether the middle frequency is
        # recorded during the triggered timeframe.
        frame = Frame.Frame()

        frame.label = is_triggered(i + freq_size / 2, tp_table, sample_rate)
        frame.data = device_data_pd.iloc[i:i + freq_size]

        list_of_dataframes.append(frame)

    # return all but the last frame, because it is not complete
    return list_of_dataframes[:-1]


# finds the start of trigger point and converts it to frequency and takes the frame_size (in seconds) and cuts each
# side into a dataframe.
# this is used to find peaks locally in EMG data.
def cut_frames(tp_table: pd.DataFrame, tt_column: str, data: pd.DataFrame,
               dataset: Dataset, frame_size: float = 2.) -> ([Frame], Dataset):
    list_of_trigger_frames = []
    indices_to_delete = []

    for i, row in tp_table.iterrows():
        start = int(row[tt_column].total_seconds() * dataset.sample_rate - frame_size * dataset.sample_rate)
        end = int(row[tt_column].total_seconds() * dataset.sample_rate + frame_size * dataset.sample_rate)
        frame = Frame.Frame()
        frame.data = dataset.data_device1.iloc[start:end]
        frame.label = 1  # indicates EMG
        frame.timestamp = row

        frame.filtered_data = data.iloc[start:end]
        frame.filtered_data = frame.filtered_data.reset_index(drop=True)
        indices_to_delete.append([start, end])
        list_of_trigger_frames.append(frame)

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        dataset.data_device1 = dataset.data_device1.drop(dataset.data_device1.index[indices[0]:indices[1]])
        data = data.drop(data.index[indices[0]:indices[1]])

    return list_of_trigger_frames, data, dataset


def slice_and_label_idle_frames(data: pd.DataFrame, filtered_data: pd.DataFrame, frame_size: int = 2, freq: int = 1200) -> [Frame]:
    list_of_frames = []
    frame_sz = frame_size * freq
    i = 0
    while i < len(data) and i + frame_sz < len(data):
        cutout = abs(data.index[i] - data.index[i + frame_sz]) == frame_sz
        if cutout:
            frame = Frame.Frame()
            frame.data = data.iloc[i:i + frame_sz]
            frame.label = 0  # indicates no EMG peak / no MRCP should be present
            frame.filtered_data = filtered_data.iloc[i:i + frame_sz]
            frame.filtered_data = frame.filtered_data.reset_index(drop=True)
            list_of_frames.append(frame)
            i += frame_sz
        else:
            i += 1

    return list_of_frames


def data_distribution(labelled_data_lst: [Frame]) -> {}:
    triggered = 0

    for frame in labelled_data_lst:
        if frame.label == 1:
            triggered += 1
    #  counter = collections.Counter(features)

    idle = len(labelled_data_lst) - triggered
    return {
        'triggered': triggered,
        'idle': idle,
        'expected_triggered_percent': int(triggered / (triggered + idle) * 100)
    }


def create_uniform_distribution(data_list: [Frame]) -> [Frame]:
    # returns the dataset with equal amount of samples, chosen by the least represented feature.
    features = []

    for frame in data_list:
        features.append(frame.label)

    counter = collections.Counter(features)
    least_represented_feature = 99999999999
    feat_counter = {}
    for i in counter.keys():
        feat_counter[i] = 0
        if counter[i] < least_represented_feature:
            least_represented_feature = counter[i]

    random.shuffle(data_list)

    uniform_data_list = []
    for frame in data_list:
        if feat_counter[frame.label] < least_represented_feature:
            uniform_data_list.append(frame)
            feat_counter[frame.label] += 1

    return uniform_data_list


def z_score_normalization(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std()


def max_absolute_scaling(frame: pd.DataFrame) -> pd.DataFrame:
    return frame / frame.abs().max()


def min_max_scaling(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.min()) / (frame.max() - frame.min())
