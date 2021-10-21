import copy
import pickle

from classes import Dataset
import pandas as pd
import biosppy
from data_preprocessing.data_distribution import cut_windows, slice_and_label_idle_windows, cut_windows_for_online, \
    slice_and_label_idle_windows_for_online
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from data_preprocessing.emg_processing import emg_clustering
from data_preprocessing.filters import butter_filter
import os
from utility.file_util import create_dir
from definitions import OUTPUT_PATH


def mrcp_detection(data: Dataset, tp_table: pd.DataFrame, config, bipolar_mode: bool = False) -> (
        [pd.DataFrame], pd.DataFrame):
    EEG_CHANNELS = list(range(0, 9))
    EMG_CHANNEL = 12
    WINDOW_SIZE = 1  # seconds
    dataset = copy.deepcopy(data)

    # Filter EMG Data
    filtered_data = pd.DataFrame()
    if bipolar_mode:
        bipolar_emg = abs(data.data_device1[EMG_CHANNEL] - data.data_device1[EMG_CHANNEL + 1])
        filtered_data[EMG_CHANNEL] = butter_filter(data=bipolar_emg,
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )
    else:
        filtered_data[EMG_CHANNEL] = butter_filter(data=dataset.data_device1[EMG_CHANNEL],
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )

    # Find onsets based on the filtered data
    onsets, = biosppy.signals.emg.find_onsets(signal=filtered_data[EMG_CHANNEL].to_numpy(),
                                              sampling_rate=dataset.sample_rate,
                                              )

    # Group onsets based on time
    emg_clusters = emg_clustering(emg_data=filtered_data[EMG_CHANNEL],
                                  onsets=onsets,
                                  freq=dataset.sample_rate,
                                  peaks_to_find=len(tp_table),
                                  )

    # Filter EEG channels with a bandpass filter
    for i in EEG_CHANNELS:
        filtered_data[i] = butter_filter(data=dataset.data_device1[i],
                                         order=config['eeg_order'],
                                         cutoff=config['eeg_cutoff'],
                                         btype=config['eeg_btype']
                                         )

    # Reshape filtered_data frame so EMG column is not first
    filtered_data = filtered_data.reindex(sorted(filtered_data.columns), axis=1)

    # Update trigger table and save filtered data
    columns = ['emg_start', 'emg_peak', 'emg_end']
    tp_table[columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset.sample_rate)
    data.filtered_data = filtered_data

    # Cut windows based on aggregation strategy and window size
    windows, filtered_data, dataset = cut_windows(tp_table=tp_table,
                                                  tt_column=config['aggregate_strategy'],
                                                  data=filtered_data,
                                                  dataset=dataset,
                                                  window_size=WINDOW_SIZE
                                                  )
    # Cut the the remaining data
    windows.extend(slice_and_label_idle_windows(data=dataset.data_device1,
                                                filtered_data=filtered_data,
                                                window_size=WINDOW_SIZE,
                                                freq=dataset.sample_rate))

    # Update information for each window in regards to ids, filter types, and extract features
    filter_type_df = pd.DataFrame(columns=[EMG_CHANNEL], data=[config['emg_btype']])
    filter_type_df[EEG_CHANNELS] = [config['eeg_btype']] * len(EEG_CHANNELS)
    filter_type_df = filter_type_df.reindex(sorted(filter_type_df.columns), axis=1)

    for i in range(0, len(windows)):
        windows[i].update_filter_type(filter_type_df)
        windows[i].num_id = i
        windows[i].aggregate_strategy = config['aggregate_strategy']
        windows[i].extract_features()

    return windows, tp_table


def emg_peaks_freq_to_datetime(emg_peaks, freq: int):
    for i in range(0, len(emg_peaks)):
        for j in range(0, len(emg_peaks[i])):
            emg_peaks[i][j] = convert_freq_to_datetime(emg_peaks[i][j], freq)

    return emg_peaks


def save_index_list(index):
    path = os.path.join(OUTPUT_PATH, 'online')
    create_dir(path, recursive=True)
    filename = os.path.join(path, 'index')
    pickle.dump(index, open(filename, 'wb'))


def load_index_list():
    path = os.path.join(OUTPUT_PATH, 'online', 'index')
    return (pickle.load(open(path, 'rb'))).tolist()


def pair_index_list(index):
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


def mrcp_detection_for_online_use(data: Dataset, tp_table: pd.DataFrame, config, bipolar_mode: bool = False) -> (
        [pd.DataFrame], pd.DataFrame):
    EEG_CHANNELS = list(range(0, 9))
    EMG_CHANNEL = 12
    WINDOW_SIZE = 1  # seconds
    dataset = copy.deepcopy(data)

    # Filter EMG Data
    filtered_data = pd.DataFrame()
    if bipolar_mode:
        bipolar_emg = abs(data.data_device1[EMG_CHANNEL] - data.data_device1[EMG_CHANNEL + 1])
        filtered_data[EMG_CHANNEL] = butter_filter(data=bipolar_emg,
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )
    else:
        filtered_data[EMG_CHANNEL] = butter_filter(data=dataset.data_device1[EMG_CHANNEL],
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )

    # Find onsets based on the filtered data
    onsets, = biosppy.signals.emg.find_onsets(signal=filtered_data[EMG_CHANNEL].to_numpy(),
                                              sampling_rate=dataset.sample_rate,
                                              )

    # Group onsets based on time
    emg_clusters = emg_clustering(emg_data=filtered_data[EMG_CHANNEL],
                                  onsets=onsets,
                                  freq=dataset.sample_rate,
                                  peaks_to_find=len(tp_table),
                                  )

    # Update trigger table and save filtered data
    columns = ['emg_start', 'emg_peak', 'emg_end']
    tp_table[columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset.sample_rate)
    data.filtered_data = filtered_data

    # Cut windows based on aggregation strategy and window size
    windows, dataset = cut_windows_for_online(tp_table=tp_table,
                                                      tt_column=config['aggregate_strategy'],
                                                      dataset=dataset,
                                                      window_size=WINDOW_SIZE
                                                      )
    # Cut the the remaining data
    windows.extend(slice_and_label_idle_windows_for_online(data=dataset.data_device1,
                                                            window_size=WINDOW_SIZE,
                                                            freq=dataset.sample_rate))

    # Update information for each window in regards to ids, filter types, and extract features
    filter_type_df = pd.DataFrame(columns=[EMG_CHANNEL], data=[config['emg_btype']])
    filter_type_df[EEG_CHANNELS] = [config['eeg_btype']] * len(EEG_CHANNELS)
    filter_type_df = filter_type_df.reindex(sorted(filter_type_df.columns), axis=1)

    index_list = data.data_device1.index.difference(dataset.data_device1.index)
    save_index_list(index_list)

    for i in range(0, len(windows)):
        windows[i].update_filter_type(filter_type_df)
        for channel in EEG_CHANNELS:
            windows[i].filter(butter_filter,
                              channel,
                              order=config['eeg_order'],
                              cutoff=config['eeg_cutoff'],
                              btype=config['eeg_btype'],
                              freq=dataset.sample_rate
                              )
        windows[i].num_id = i
        windows[i].aggregate_strategy = config['aggregate_strategy']
        windows[i].extract_features()

    return windows, tp_table
