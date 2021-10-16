import copy

from classes import Dataset
import pandas as pd
import biosppy
from data_preprocessing.data_distribution import cut_windows, slice_and_label_idle_windows
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from data_preprocessing.emg_processing import emg_clustering
from data_preprocessing.filters import butter_filter


def mrcp_detection(data: Dataset, tp_table: pd.DataFrame, config, bipolar_mode: bool = False) -> (
[pd.DataFrame], pd.DataFrame):
    EEG_CHANNELS = list(range(0, 10))
    EMG_CHANNEL = 12
    FRAME_SIZE = 1  # seconds
    dataset = copy.deepcopy(data)

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

    onsets, = biosppy.signals.emg.find_onsets(signal=filtered_data[EMG_CHANNEL].to_numpy(),
                                              sampling_rate=dataset.sample_rate,
                                              )

    emg_clusters = emg_clustering(emg_data=filtered_data[EMG_CHANNEL],
                                  onsets=onsets,
                                  freq=dataset.sample_rate,
                                  peaks_to_find=len(tp_table),
                                  )

    for i in EEG_CHANNELS:
        filtered_data[i] = butter_filter(data=dataset.data_device1[i],
                                         order=config['eeg_order'],
                                         cutoff=config['eeg_cutoff'],
                                         btype=config['eeg_btype']
                                         )

    filtered_data = filtered_data.reindex(sorted(filtered_data.columns), axis=1)

    columns = ['emg_start', 'emg_peak', 'emg_end']
    tp_table[columns] = emg_peaks_freq_to_datetime(emg_clusters, dataset.sample_rate)
    data.filtered_data = filtered_data

    windows, filtered_data, dataset = cut_windows(tp_table=tp_table,
                                                  tt_column=config['aggregate_strategy'],
                                                  data=filtered_data,
                                                  dataset=dataset,
                                                  window_size=FRAME_SIZE
                                                  )

    windows.extend(slice_and_label_idle_windows(data=dataset.data_device1,
                                                filtered_data=filtered_data,
                                                window_size=FRAME_SIZE,
                                                freq=dataset.sample_rate))

    filter_type_df = pd.DataFrame(columns=[EMG_CHANNEL], data=[config['emg_btype']])
    filter_type_df[EEG_CHANNELS] = [config['eeg_btype']] * len(EEG_CHANNELS)
    filter_type_df = filter_type_df.reindex(sorted(filter_type_df.columns), axis=1)

    for i in range(0, len(windows)):
        windows[i].update_filter_type(filter_type_df)
        windows[i].num_id = i

    return windows, tp_table


def emg_peaks_freq_to_datetime(emg_peaks, freq: int):
    for i in range(0, len(emg_peaks)):
        for j in range(0, len(emg_peaks[i])):
            emg_peaks[i][j] = convert_freq_to_datetime(emg_peaks[i][j], freq)

    return emg_peaks
