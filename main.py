import pandas as pd
import glob
import scipy.io
import json
import os

# Data preprocessing imports
from data_preprocessing.data_distribution import create_uniform_distribution
from data_preprocessing.data_shift import shift_data
from data_preprocessing.optimize_windows import optimize_average_minimum, remove_worst_windows, find_best_config_params, \
    prune_poor_quality_samples
from data_preprocessing.fourier_transform import fourier_transform_listof_datawindows, \
    fourier_transform_single_datawindow
from data_preprocessing.mrcp_detection import mrcp_detection
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date, convert_freq_to_datetime
from data_preprocessing.trigger_points import covert_trigger_points_to_pd, trigger_time_table
from data_preprocessing.train_test_split import train_test_split_data

# Data visualization imports
from data_visualization.average_channels import find_usable_emg, average_channel, plot_average_channels

# Training/Classification imports
from data_training.LGBM.lgbm_prediction import lgbm_classifier
from data_training.SVM.svm_prediction import svm_classifier
from data_training.KNN.knn_prediction import knn_classifier
from data_training.LDA.lda_prediction import lda_classifier

# Logging imports
import logging
from utility.logger import get_logger
from utility.save_and_load import save_train_test_split, load_train_test_split
from utility.pdf_creation import save_results_to_pdf


from definitions import DATASET_PATH, OUTPUT_PATH
from classes import Dataset, Window

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
pd.set_option("display.max_rows", None, "display.max_columns", None)  # Datawindow print settings
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

    dataset = init(selected_cue_set=config['id'])

    # Shift Data to remove startup
    dataset = shift_data(freq=80000, dataset=dataset)

    # Create table containing information when trigger points were shown/removed
    trigger_table = trigger_time_table(dataset.TriggerPoint, dataset.time_start_device1)

    if script_params['run_mrcp_detection']:
        # Perform MRCP Detection and update trigger_table with EMG timestamps
        windows, trigger_table = mrcp_detection(data=dataset, tp_table=trigger_table, config=config)

        # Plot all filtered channels (0-9 and 12) together with the raw data
        dataset.plot()

        # Remove poor quality samples based on heuristic and score
        prune_poor_quality_samples(windows, trigger_table, config, remove=10, method=remove_worst_windows)

        # Plot Average and Individual Frames
        avg_windows = average_channel(windows)
        plot_average_channels(avg_windows, save_fig=False, overwrite=True)

        for window in windows:
            window.plot()

        # Create distribution for training and dividing into train and test set
        uniform_data = create_uniform_distribution(windows)
        train_data, test_data = train_test_split_data(uniform_data, split_per=20)

        # save_train_test_split(train_data, test_data, dir_name='EEG')

    if script_params['run_classification']:

        train_data, test_data = load_train_test_split(dir_name='EEG')

        feature = 'features'
        knn_score = knn_classifier(train_data, test_data, features=feature)
        svm_score = svm_classifier(train_data, test_data, features=feature)
        lda_score = lda_classifier(train_data, test_data, features=feature)

        results = {
            'KNN_results': knn_score,
            'SVM_results': svm_score,
            'LDA_results': lda_score
        }

        # Writes the test and train window plots + classifier score tables to pdf file
        save_results_to_pdf(train_data, test_data, results, file_name='result_overview.pdf')


