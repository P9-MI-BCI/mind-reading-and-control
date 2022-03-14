import operator
import os
import copy
import pickle
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
from classes import Dataset
from data_preprocessing.data_distribution import cut_mrcp_windows, cut_and_label_idle_windows, \
    cut_mrcp_windows_rest_movement_phase, cut_mrcp_windows_calibration
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from data_preprocessing.emg_processing import onset_detection
from data_preprocessing.filters import butter_filter
from data_visualization.visualize_windows import visualize_window_all_channels
from utility.file_util import create_dir
from definitions import OUTPUT_PATH, CONFIG_PATH
import json
from utility.logger import get_logger
from scipy.special import softmax
from scipy.stats import zscore


def emg_peaks_freq_to_datetime(emg_peaks, freq: int):
    for i in range(0, len(emg_peaks)):
        for j in range(0, len(emg_peaks[i])):
            emg_peaks[i][j] = convert_freq_to_datetime(emg_peaks[i][j], freq)

    return emg_peaks


def update_config(config):
    with open(CONFIG_PATH, 'r') as readfile:
        old_config = json.load(readfile)

    old_config[config.cue_set_name] = config

    with open(CONFIG_PATH, 'w') as writefile:
        json.dump(old_config, writefile, indent=4, sort_keys=True)


# Writes the indexes of windows with mrcp to a file
def save_index_list(index: [int], config):
    path = os.path.join(OUTPUT_PATH, config.cue_set_name)
    create_dir(path, recursive=True)
    filename = os.path.join(path, 'index')
    pickle.dump(index, open(filename, 'wb'))
    config.index_timestamp = time.time()
    config.index = filename


def load_index_list(path) -> [int]:
    return (pickle.load(open(path, 'rb'))).tolist()


# converts index list to a pair_index list where it holds pairs of (start, end) of each range
def pair_index_list(index: [int]):
    pair_indexes = []
    start, current, end = index[0], index[0], 0
    for i in range(1, len(index) - 1):
        if index[i] - 1 == current:
            current = index[i]
        else:
            end = index[i - 1]
            pair_indexes.append((start, end))

            start = index[i]
            current = index[i]

    return pair_indexes


def channel_weights_calculation(average_channels):
    negativity_distance = [0] * len(average_channels)
    center = int(len(average_channels[0].data) / 2)
    for channel in range(0, len(average_channels)):
        negativity_distance[channel] = abs(average_channels[channel].data.idxmin() - center)

    normalized = zscore(negativity_distance)
    return softmax(normalized)


# todo DEPRECATED
def mrcp_detection_for_calibration(data: Dataset, config, perfect_centering: bool, weights=None) -> (
        [pd.DataFrame], pd.DataFrame):
    dataset_copy = copy.deepcopy(data)

    # Find EMG onsets and group onsets based on time
    emg_clusters, filtered_data = onset_detection(dataset_copy, config)

    get_logger().info(f'Found {len(emg_clusters)} potential EMG onsets.')
    # Filter EEG channels with a bandpass filter
    for i in config.EEG_Channels:
        filtered_data[i] = butter_filter(data=dataset_copy.data_device1[i],
                                         order=config.eeg_order,
                                         cutoff=config.eeg_cutoff,
                                         btype=config.eeg_btype
                                         )


    # Reshape filtered_data frame so EMG column is not first
    filtered_data = filtered_data.reindex(sorted(filtered_data.columns), axis=1)

    scaler = StandardScaler()
    scaler.fit(filtered_data[config.EEG_Channels])

    filtered_data[config.EEG_Channels] = scaler.transform(filtered_data[config.EEG_Channels])

    # Update trigger table and save filtered data
    data.filtered_data = filtered_data

    tp_table = pd.DataFrame(columns=['emg_start', 'emg_peak', 'emg_end'])
    tp_table[tp_table.columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset_copy.sample_rate)
    # Cut windows based on aggregation strategy and window size
    windows = cut_mrcp_windows_calibration(tp_table=tp_table,
                                           tt_column=config.aggregate_strategy,
                                           filtered_data=filtered_data,
                                           dataset=dataset_copy,
                                           window_size=config.window_size,
                                           weights=weights,
                                           perfect_centering=perfect_centering,
                                           multiple_windows=False
                                           )

    # Cut the the remaining data
    windows.extend(cut_and_label_idle_windows(data=dataset_copy.data_device1,
                                              filtered_data=filtered_data,
                                              window_size=config.window_size,
                                              freq=dataset_copy.sample_rate,
                                              ))

    # Update information for each window in regards to ids, filter types, and extract features
    filter_type_df = pd.DataFrame(columns=[config.EMG_Channel], data=[config.emg_btype])
    filter_type_df[config.EEG_Channels] = [config.eeg_btype] * len(config.EEG_Channels)
    filter_type_df = filter_type_df.reindex(sorted(filter_type_df.columns), axis=1)

    # this code will not work if the config does not have a cue_set_name field set.
    # index_list = data.data_device1.index.difference(dataset_copy.data_device1.index)
    # save_index_list(index_list, config)

    # sort mrcp windows first, implicit knowledge used for other functionality later in code
    windows.sort(key=operator.attrgetter('label'), reverse=True)

    # Updates each window with various information
    for i, window in enumerate(windows):
        window.update_filter_type(filter_type_df)
        window.aggregate_strategy = config.aggregate_strategy
        window.extract_features()
        window.create_feature_vector()

    return windows


