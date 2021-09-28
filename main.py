import pandas as pd
import glob
import os
import scipy.io
import matplotlib.pyplot as plt
from data_preprocessing.data_distribution import aggregate_data, create_uniform_distribution, z_score_normalization, \
    max_absolute_scaling, min_max_scaling, aggregate_trigger_points_for_emg_peak, slice_and_label_idle_frames
from data_preprocessing.emg_processing import find_emg_peaks
from data_preprocessing.fourier_transform import fourier_transform_listof_dataframes, fourier_transform_single_dataframe
from definitions import DATASET_PATH
from classes import Dataset
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date, convert_freq_to_datetime
from data_preprocessing.trigger_points import covert_trigger_points_to_pd, trigger_time_table
from data_preprocessing.train_test_split import train_test_split_data
from data_training.KNN.knn_prediction import knn_classifier_all_channels

# Logging imports
import logging
from utility.logger import get_logger
from utility.save_and_load import save_train_test_split, load_train_test_split

get_logger().setLevel(logging.INFO)
pd.set_option("display.max_rows", None, "display.max_columns", None)


def init(selected_cue_set=0):
    cue_sets = []

    for file in glob.glob(DATASET_PATH, recursive=True):
        cue_sets.append(scipy.io.loadmat(file))

    cue_set = cue_sets[selected_cue_set]

    dataset = Dataset.Dataset()

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
    dataset.data_device1 = pd.DataFrame(cue_set['data_device1'][7:])
    dataset.time_axis_all_device1 = pd.DataFrame(cue_set['time_axis_all_device1'])

    return dataset


def init_emg(data):
    emg_peaks = find_emg_peaks(data, type='highest')

    for i in range(0, len(emg_peaks)):
        emg_peaks[i] = convert_freq_to_datetime(emg_peaks[i], data.sample_rate)

    trigger_table['emg_peaks'] = emg_peaks

    emg_frames, data = aggregate_trigger_points_for_emg_peak(trigger_table, 'emg_peaks', data, frame_size=2)

    emg_frames.extend(slice_and_label_idle_frames(data.data_device1))
    return emg_frames


if __name__ == '__main__':
    data = init(selected_cue_set=0)

    trigger_table = trigger_time_table(data.TriggerPoint, data.time_start_device1)

    # normalization / scaling techniques
    # data.data_device1 = fourier_transform_single_dataframe(data.data_device1)
    # data.data_device1 = z_score_normalization(data.data_device1)
    # data.data_device1 = max_absolute_scaling(data.data_device1)
    # data.data_device1 = min_max_scaling(data.data_device1)
    emg_frames = init_emg(data)

    # labelled_data = aggregate_data(data.data_device1, 100, trigger_table, sample_rate=data.sample_rate)
    uniform_data = create_uniform_distribution(emg_frames)

    # uniform_data = fourier_transform_listof_dataframes(uniform_data)
    train_data, test_data = train_test_split_data(uniform_data, split_per=20)
    save_train_test_split(train_data, test_data, 'emg_uniform_four_shuffled')

    train_data, test_data = load_train_test_split('emg_uniform_four_shuffled')

    score = knn_classifier_all_channels(train_data, test_data)
    print(score)
