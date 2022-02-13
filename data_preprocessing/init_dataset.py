import glob
import pandas as pd
import scipy.io
import mne
import numpy as np
import os
from classes.Dataset import Dataset
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date
from data_preprocessing.filters import butter_filter
from data_preprocessing.trigger_points import covert_trigger_points_to_pd
from definitions import DATASET_PATH


# Takes care of loading in the dataset into our Dataset class
def init(selected_cue_set: int = 0):
    cue_sets = []

    for file in glob.glob(DATASET_PATH, recursive=True):
        cue_sets.append(scipy.io.loadmat(file))

    cue_set = cue_sets[selected_cue_set]

    dataset = Dataset()

    try:
        dataset.handle_arrow_rand = cue_set['handle_arrow_rand'][0]
        dataset.no_movements = cue_set['no_movements'][0][0]
        dataset.time_cue_on = convert_mat_date_to_python_date(cue_set['time_cue_on'])
        dataset.time_cue_off = convert_mat_date_to_python_date(cue_set['time_cue_off'])
        dataset.TriggerPoint = covert_trigger_points_to_pd(cue_set['TriggerPoint'])
        dataset.delay_T1 = cue_set['delay_T1'][0][0]
        dataset.delay_random_T1 = cue_set['delay_random_T1'][0][0]
        dataset.delay_T2 = cue_set['delay_T2'][0][0]
        dataset.sample_rate = cue_set['sample_rate'][0][0]
        dataset.time_window = cue_set['time_window'][0][0]
        dataset.no_time_windows = cue_set['no_time_windows'][0][0]
        dataset.filter_code_eeg = cue_set['filter_code_eeg'][0][0]
        dataset.time_start_device1 = convert_mat_date_to_python_date(cue_set['time_start_device1'])
        dataset.time_after_first_window = convert_mat_date_to_python_date(cue_set['time_after_first_window'])
        dataset.time_after_last_window = convert_mat_date_to_python_date(cue_set['time_after_last_window'])
        dataset.time_stop_device1 = convert_mat_date_to_python_date(cue_set['time_stop_device1'])
        dataset.time_axis_all_device1 = pd.DataFrame(cue_set['time_axis_all_device1'])
    except:
        dataset.sample_rate = 1200
        print('Missing values')

    dataset.data_device1 = pd.DataFrame(cue_set['data_device1'])

    return dataset


def load_data_to_mne_epoch(dataset_path, config):
    data = []

    for file in glob.glob(dataset_path, recursive=True):
        data.append(scipy.io.loadmat(file))

    if len(data) == 1:
        data = data[0]
    else:
        # todo handle multiple training datasets
        return 0

    filtered_data = pd.DataFrame()
    for i in config.EEG_Channels:
        filtered_data[i] = butter_filter(data=data['data_device1'][:,i],
                                         order=config.eeg_order,
                                         cutoff=config.eeg_cutoff,
                                         btype=config.eeg_btype
                                         )

    ch_names = ['T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8']
    ch_types = ['eeg'] * len(config.EEG_Channels)

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=config.sample_rate)
    info.set_montage('standard_1020')

    raw_data = mne.io.RawArray(np.transpose(filtered_data[config.EEG_Channels]), info)
    raw_data.plot()
    raw_data.plot_sensors(kind='3d', ch_type='all')
    return raw_data


def get_datasets(subject_id: int, config):
    dwell_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'dwell_tuning/*')
    online_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'online_test/*')
    training_p = os.path.join(DATASET_PATH, f'subject_{subject_id}', 'training/*')

    training_dataset = load_data_to_mne_epoch(training_p, config)
    dwell_dataset = load_data_to_mne_epoch(dwell_p, config)
    online_dataset = load_data_to_mne_epoch(online_p, config)
    # training_dataset = load_data_to_mne_epoch(training_p, config)

    return dwell_dataset, online_dataset, training_dataset