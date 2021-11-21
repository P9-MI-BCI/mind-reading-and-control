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
from data_preprocessing.emg_processing import onset_detection, onset_threshold_detection, onset_detection_calibration
from data_preprocessing.eog_detection import blink_detection
from data_preprocessing.filters import butter_filter
from utility.file_util import create_dir
from definitions import OUTPUT_PATH, CONFIG_PATH
import json
from utility.logger import get_logger


def mrcp_detection(data: Dataset, tp_table: pd.DataFrame, config, bipolar_mode: bool = False,
                   calibration: bool = False) -> (
        [pd.DataFrame], pd.DataFrame):
    EEG_CHANNELS = list(range(0, 9))
    EMG_CHANNEL = 12
    dataset_copy = copy.deepcopy(data)

    # Find EMG onsets and group onsets based on time
    emg_clusters, filtered_data = onset_detection(dataset_copy, tp_table, config, bipolar_mode)

    # Find EMG onsets with a static threshold and group onsets based on time
    # emg_clusters, filtered_data = onset_threshold_detection(dataset=dataset_copy, tp_table=tp_table, config=config)

    # Filter EEG channels with a bandpass filter
    for i in EEG_CHANNELS:
        filtered_data[i] = butter_filter(data=dataset_copy.data_device1[i],
                                         order=config.eeg_order,
                                         cutoff=config.eeg_cutoff,
                                         btype=config.eeg_btype
                                         )

    # Reshape filtered_data frame so EMG column is not first
    filtered_data = filtered_data.reindex(sorted(filtered_data.columns), axis=1)

    # # Update trigger table and save filtered data
    # columns = ['emg_start', 'emg_peak', 'emg_end']
    # tp_table[columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset_copy.sample_rate)
    # tp_table = fix_time_table(tp_table)
    tp_table = pd.DataFrame(columns=['emg_start', 'emg_peak', 'emg_end'])
    tp_table[tp_table.columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset_copy.sample_rate)

    data.filtered_data = filtered_data

    # Cut windows based on aggregation strategy and window size
    windows = cut_mrcp_windows_rest_movement_phase(tp_table=tp_table,
                                                   tt_column=config.aggregate_strategy,
                                                   filtered_data=filtered_data,
                                                   dataset=dataset_copy,
                                                   window_size=config.window_size,
                                                   perfect_centering=False,
                                                   multiple_windows=False
                                                   )

    # Cut the the remaining data
    windows.extend(cut_and_label_idle_windows(data=dataset_copy.data_device1,
                                              filtered_data=filtered_data,
                                              window_size=config.window_size,
                                              freq=dataset_copy.sample_rate,
                                              ))

    # Update information for each window in regards to ids, filter types, and extract features
    filter_type_df = pd.DataFrame(columns=[EMG_CHANNEL], data=[config.emg_btype])
    filter_type_df[EEG_CHANNELS] = [config.eeg_btype] * len(EEG_CHANNELS)
    filter_type_df = filter_type_df.reindex(sorted(filter_type_df.columns), axis=1)

    if not calibration:
        index_list = data.data_device1.index.difference(dataset_copy.data_device1.index)
        save_index_list(index_list, config)

    # Find frequencies of all detected blinks from EOG channel 9
    blinks = blink_detection(data=data.data_device1, sample_rate=data.sample_rate)

    # sort mrcp windows first, implicit knowledge used for other functionality later in code
    windows.sort(key=operator.attrgetter('label'), reverse=True)

    # Updates each window with various information
    for i, window in enumerate(windows):
        window.update_filter_type(filter_type_df)
        window.aggregate_strategy = config.aggregate_strategy
        window.extract_features()
        window.blink_detection(blinks)
        window.create_feature_vector()

    if not calibration:
        update_config(config)

    return windows, tp_table


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


def mrcp_detection_for_calibration(data: Dataset, config, input_peaks, bipolar_mode: bool = False) -> (
        [pd.DataFrame], pd.DataFrame):
    dataset_copy = copy.deepcopy(data)

    # Find EMG onsets and group onsets based on time
    emg_clusters, filtered_data = onset_detection_calibration(dataset_copy, config, input_peaks=input_peaks)

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
                                           perfect_centering=False,
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

    # Find frequencies of all detected blinks from EOG channel 9
    blinks = blink_detection(data=data.data_device1, sample_rate=data.sample_rate)

    # sort mrcp windows first, implicit knowledge used for other functionality later in code
    windows.sort(key=operator.attrgetter('label'), reverse=True)

    # Updates each window with various information
    for i, window in enumerate(windows):
        window.update_filter_type(filter_type_df)
        window.aggregate_strategy = config.aggregate_strategy
        window.extract_features()
        window.blink_detection(blinks)
        window.create_feature_vector()

    return windows, scaler


def surrogate_channels(data: pd.DataFrame):
    surrogate_channel = []

    for i, row in data.iterrows():
        temp = []
        for col in data.columns:
            temp.append(surrogate(row[col]))

        surrogate_channel.append(sum(temp))

    return surrogate_channel


def surrogate(y):
    return -(1 / (y - 1))


def fix_time_table(trigger_table: pd.DataFrame) -> pd.DataFrame:
    columns = ['emg_start', 'emg_peak', 'emg_end']
    tp_cols = ['tp_start', 'tp_end']

    fixed_list = []
    counter = 0
    # checks if next timestamp is closer and applies that instead.
    for i in range(len(trigger_table) - 1):
        time_diff = abs(trigger_table.iloc[i]['emg_start'] - trigger_table.iloc[i]['tp_start'])
        if time_diff > abs(trigger_table.iloc[i + 1]['emg_start'] - trigger_table.iloc[i]['tp_start']):
            temp = trigger_table.iloc[i][tp_cols]
            temp = temp.append(trigger_table.iloc[i + 1][columns])
            fixed_list.append(temp)
            counter -= 1
        else:
            fixed_list.append(trigger_table.iloc[i])
        counter += 1

    if counter == len(trigger_table) - 1:
        fixed_list.append(trigger_table.iloc[-1])
    fixed_tp_table = pd.DataFrame(fixed_list)
    fixed_tp_table = fixed_tp_table.reset_index(drop=True)

    return fixed_tp_table
