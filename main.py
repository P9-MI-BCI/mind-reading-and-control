import pandas as pd
import glob
import os
import scipy.io

from data_preprocessing.data_distribution import aggregate_data, create_uniform_distribution, z_score_normalization, \
    max_absolute_scaling, min_max_scaling
from definitions import DATASET_PATH
from classes import Dataset
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date
from data_preprocessing.trigger_points import covert_trigger_points_to_pd, trigger_time_table
from data_preprocessing.train_test_split import train_test_split_data
from data_training.KNN.knn_prediction import knn_classifier_all_channels

# Logging imports
import logging
from utility.logger import get_logger
get_logger().setLevel(logging.WARNING)


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
    dataset.data_device1 = pd.DataFrame(cue_set['data_device1'][15:])
    dataset.time_axis_all_device1 = pd.DataFrame(cue_set['time_axis_all_device1'])

    return dataset


if __name__ == '__main__':
    data = init(selected_cue_set=0)

    trigger_table = trigger_time_table(data.TriggerPoint, data.time_start_device1)

    # normalization / scaling techniques
    # data.data_device1 = z_score_normalization(data.data_device1)
    # data.data_device1 = max_absolute_scaling(data.data_device1)
    # data.data_device1 = min_max_scaling(data.data_device1)

    labelled_data = aggregate_data(data.data_device1, 100, trigger_table, sample_rate=data.sample_rate)
    uniform_data = create_uniform_distribution(labelled_data)

    train_data, test_data = train_test_split_data(uniform_data, split_per=20)

    score = knn_classifier_all_channels(train_data, test_data)
