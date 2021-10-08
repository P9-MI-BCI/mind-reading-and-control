import biosppy
import pandas as pd
import glob
import os
import scipy.io
import matplotlib.pyplot as plt
from data_preprocessing.data_distribution import aggregate_data, create_uniform_distribution, z_score_normalization, \
    max_absolute_scaling, min_max_scaling, aggregate_trigger_points_for_emg_peak, slice_and_label_idle_frames
from data_preprocessing.emg_processing import find_emg_peaks
from data_preprocessing.filters import butter_filter
from data_preprocessing.fourier_transform import fourier_transform_listof_dataframes, fourier_transform_single_dataframe
from data_training.LGBM.lgbm_prediction import lgbm_classifier
from data_training.SVM.svm_prediction import svm_classifier
from data_visualization.average_channels import find_usable_emg, average_channel, plot_average_channels
from data_visualization.timestamp_visualization import visualize_frame
from definitions import DATASET_PATH, OUTPUT_PATH
from classes import Dataset, Frame
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date, convert_freq_to_datetime
from data_preprocessing.trigger_points import covert_trigger_points_to_pd, trigger_time_table
from data_preprocessing.train_test_split import train_test_split_data
from data_training.KNN.knn_prediction import knn_classifier
import copy
import json
import statistics
# Logging imports
import logging
from utility.logger import get_logger
from utility.save_and_load import save_train_test_split, load_train_test_split
from data_visualization.eeg_plotting import plot_eeg

'''CONFIGURATION'''
get_logger().setLevel(logging.INFO)  # Set logging level
pd.set_option("display.max_rows", None, "display.max_columns", None)  # Dataframe print settings
with open('config.json') as config_file:
    config = json.load(config_file)['cue_set1']  # Choose config


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
    dataset.data_device1 = pd.DataFrame(
        cue_set['data_device1'][7:])  # removes the startup values 7 freq is considered insignificant
    dataset.time_axis_all_device1 = pd.DataFrame(cue_set['time_axis_all_device1'])

    return dataset


def init_emg(dataset: Dataset, tp_table: pd.DataFrame) -> ([pd.DataFrame], pd.DataFrame):
    eeg_channels = list(range(0, 9))

    all_filtered_data = butter_filter(dataset.data_device1[config['EMG_CHANNEL']], order=config['all_butter_order'],
                                      cutoff=config['all_butter_cutoff'])

    # data_cop = copy.deepcopy(dataset)
    # types = ['sos_lowpass']
    #
    # for type in types:
    #     for i in eeg_channels:
    #         if type == 'bandpass' or type == 'sos_bandpass':
    #             data_cop.data_device1[i] = butter_filter(data_cop.data_device1[i], btype=type, order=2,
    #                                                     cutoff=[0.05, 5])
    #         elif type == 'lowpass' or type == 'sos_lowpass':
    #             data_cop.data_device1[i] = butter_filter(data_cop.data_device1[i], btype=type, order=2,
    #                                                      cutoff=[5])
    #         elif type == 'highpass' or type == 'sos_highpass':
    #             data_cop.data_device1[i] = butter_filter(data_cop.data_device1[i], btype=type, order=4,
    #                                                      cutoff=[80])
    #
    #     plot_eeg(dataset.data_device1, data_cop.data_device1, type, all=True)

    onsets, = biosppy.signals.emg.find_onsets(signal=all_filtered_data, sampling_rate=dataset.sample_rate)

    # dataset.data_device1[eeg_channels] = z_score_normalization(data.data_device1[eeg_channels])
    emg_peaks = find_emg_peaks(dataset, onsets, filtered=all_filtered_data, peaks_to_find=len(tp_table),
                               channel=config['EMG_CHANNEL'])

    for i in range(0, len(emg_peaks)):
        for j in range(0, len(emg_peaks[i])):
            emg_peaks[i][j] = convert_freq_to_datetime(emg_peaks[i][j], dataset.sample_rate)

    columns = ['emg_start', 'emg_peak', 'emg_end']
    tp_table[columns] = emg_peaks

    # data_cop = copy.deepcopy(dataset)
    # data_cop.data_device1 = pd.DataFrame(butter_filter(data_cop.data_device1[eeg_channels], order=2, cutoff=[0.05, 3], btype='bandpass', freq=1200))

    frames, dataset = aggregate_trigger_points_for_emg_peak(tp_table, 'emg_peak', dataset,
                                                            frame_size=2)

    frames.extend(slice_and_label_idle_frames(dataset.data_device1))

    # for frame in frames:
    #     frame.filter(butter_filter, eeg_channels, order=4, cutoff=[0.05], btype='lowpass', freq=1200)

    for i in eeg_channels:
        for frame in frames:
            frame.filter(butter_filter, i, btype='lowpass', order=2, cutoff=[5])


    return frames, tp_table


if __name__ == '__main__':
    data = init(selected_cue_set=config['id'])

    trigger_table = trigger_time_table(data.TriggerPoint, data.time_start_device1)

    '''normalization / scaling techniques'''
    # data.data_device1 = fourier_transform_single_dataframe(data.data_device1)
    # data.data_device1 = z_score_normalization(data.data_device1)
    # data.data_device1 = max_absolute_scaling(data.data_device1)
    # data.data_device1 = min_max_scaling(data.data_device1)

    # for i in range(0,9):
    #     plt.plot(data.data_device1[i], label=f'Channel {i + 1}')
    # plt.legend()
    # plt.show()

    emg_frames, trigger_table = init_emg(data, trigger_table)
    # valid_emg = find_usable_emg(trigger_table)
    #
    # avg = average_channel(emg_frames, valid_emg)
    # plot_average_channels(avg, save_fig=True)

    for i in range(0, len(emg_frames)):
       if emg_frames[i].label == 1:
           visualize_frame(emg_frames[i], data.sample_rate, channel=3, num=i)

    # labelled_data = aggregate_data(data.data_device1, 100, trigger_table, sample_rate=data.sample_rate)
    # uniform_data = create_uniform_distribution(emg_frames)
    #
    # # uniform_data = fourier_transform_listof_dataframes(uniform_data)
    # train_data, test_data = train_test_split_data(uniform_data, split_per=20)
    # save_train_test_split(train_data, test_data, 'emg_uniform')
    #
    # train_data, test_data = load_train_test_split('emg_uniform')
    #
    # score = knn_classifier(train_data, test_data, channels=[3, 4, 5])
    # print(score)
