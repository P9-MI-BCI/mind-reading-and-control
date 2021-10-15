import pandas as pd
import glob
import scipy.io
import os
import sys
import json

# Data preprocessing imports
from data_preprocessing.data_distribution import create_uniform_distribution
from data_preprocessing.data_shift import shift_data
from data_preprocessing.find_best_params import optimize_average_minimum, remove_worst_windows, find_best_config_params
from data_preprocessing.fourier_transform import fourier_transform_listof_datawindows, fourier_transform_single_datawindow
from data_preprocessing.mrcp_detection import mrcp_detection
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date, convert_freq_to_datetime
from data_preprocessing.trigger_points import covert_trigger_points_to_pd, trigger_time_table
from data_preprocessing.train_test_split import train_test_split_data

# Data visualization imports
from data_visualization.average_channels import find_usable_emg, average_channel, plot_average_channels
from data_visualization.timestamp_visualization import visualize_window
from data_visualization.raw_and_filtered_data import plot_raw_filtered_data

# Training/Classification imports
from data_training.LGBM.lgbm_prediction import lgbm_classifier
from data_training.SVM.svm_prediction import svm_classifier
from data_training.KNN.knn_prediction import knn_classifier

# Logging imports
import logging
from utility.logger import get_logger
from utility.save_and_load import save_train_test_split, load_train_test_split

from definitions import DATASET_PATH, OUTPUT_PATH
from classes import Dataset, Window

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # Datawindow print settings
with open('config.json') as config_file, open('script_parameters.json') as script_parameters:
    config = json.load(config_file)['cue_set0']  # Choose config
    script_params = json.load(script_parameters)  # Load script parameters


def init(selected_cue_set: int = 0):
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
    dataset.data_device1 = pd.DataFrame(cue_set['data_device1'])
    dataset.time_axis_all_device1 = pd.DataFrame(cue_set['time_axis_all_device1'])

    return dataset


if __name__ == '__main__':

    data = init(selected_cue_set=config['id'])

    # Shift Data to remove startup
    data = shift_data(freq=80000, dataset=data)

    # Create table containing information when trigger points were shown/removed
    trigger_table = trigger_time_table(data.TriggerPoint, data.time_start_device1)

    if script_params['run_mrcp_detection']:
        # Perform MRCP Detection and update trigger_table with EMG timestamps
        emg_windows, trigger_table = mrcp_detection(data=data, tp_table=trigger_table, config=config)

        # Plot all filtered channels (0-8 and 12) together with the raw data
        plot_raw_filtered_data(data=data, save_fig=False, overwrite=True)

        # Find valid emgs based on heuristic and calculate averages
        valid_emg = find_usable_emg(trigger_table, config)
        valid_emg = optimize_average_minimum(valid_emg, emg_windows, remove=8, weights=[0.2, 0.2, 1, 0.2, 0.2, 1, 0.2, 0.2, 0.2])
        # valid_emg = remove_worst_windows(valid_emg, emg_windows, remove=8, weights=[0.2, 0.2, 1, 0.2, 0.2, 1, 0.2, 0.2, 0.2])
        avg = average_channel(emg_windows, valid_emg)
        plot_average_channels(avg, save_fig=False, overwrite=True)

        # Plot individual windows
        for i in range(0, len(valid_emg)):
            visualize_window(emg_windows[valid_emg[i]], config=config, freq=data.sample_rate, channel=4, num=valid_emg[i], save_fig=False, overwrite=True)

    if script_params['run_classification']:
        uniform_data = create_uniform_distribution(emg_windows)
        train_data, test_data = train_test_split_data(uniform_data, split_per=20)
        save_train_test_split(train_data, test_data, 'eeg')

        train_data, test_data = load_train_test_split('eeg')

        score = knn_classifier(train_data, test_data)
        print(score)
