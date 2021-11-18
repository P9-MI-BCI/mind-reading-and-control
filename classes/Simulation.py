import sys
import time

import flask
import pandas as pd
from tqdm import tqdm

from classes.Dataset import Dataset
from classes.Window import Window
import numpy as np
import time
import json
from data_preprocessing.data_distribution import create_uniform_distribution

from data_preprocessing.data_shift import shift_data
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from data_preprocessing.mrcp_detection import load_index_list, pair_index_list, \
    mrcp_detection_for_calibration
from data_preprocessing.optimize_windows import optimize_channels
from data_training.measurements import accuracy, precision, recall, f1
from data_visualization.average_channels import average_channel, plot_average_channels
from utility.logger import get_logger
import datetime
import collections
from data_preprocessing.filters import butter_filter
from classes.Dict import AttrDict

# Database imports
from api import sql_create_windows_table, table_exist, truncate_table
from definitions import DB_PATH
import sqlite3

# Database configuration
connex = sqlite3.connect(DB_PATH)  # Opens file if exists, else creates file
cur = connex.cursor()
cur.execute(table_exist('Windows'))
if not cur.fetchone()[0] == 1:  # Checks if Windows table exist, else create new Windows table
    cur.executescript(sql_create_windows_table())
# cur.execute(truncate_table('Windows'))  # Removes all records in table from last run
cur.close()



TIME_PENALTY = 60  # 50 ms
TIME_TUNER = 1  # 0.90  # has to be adjusted to emulate real time properly.


class Simulation:

    def __init__(self, config, dataset: Dataset = 0, calibration_dataset: Dataset = 0, real_time: bool = False):
        self.time = time.time()
        self.iteration = 0
        self.real_time = real_time
        self.sliding_window = Window()
        self.model = None
        self.metrics = None  # maybe implement metrics class
        self.prev_pred_buffer = [0] * 5
        self.mrcp_detected = False
        self.freeze_flag = False
        self.freeze_counter = 0
        self.data_buffer_flag = True
        self.predictions = []
        self.prediction_frequency = []
        self.true_labels = []
        self.score = {}
        self.normalization = None

        if config:
            self.window_size = config.window_size
            self.buffer_size = self.window_size * config.buffer_size
            self.step_size = config.step_size
            self.allowed_time = self.step_size
            self.EEG_channels = config.EEG_Channels
            self.data_buffer = pd.DataFrame(columns=self.EEG_channels)

            self.filtering = config.filtering
            self.EEG_order = config.eeg_order
            self.EEG_cutoff = config.eeg_cutoff
            self.EEG_btype = config.eeg_btype

            self.cue_set = config.id
            self.start_time = config.start_time
            self.freeze_time = config.freeze_time
            if config.index is not None:
                self.index = config.index
                self.index_time = config.index_timestamp
                self._build_index()
                self.index_position = 0
            else:
                self.index = None

        if dataset:
            self.dataset = None
            self.mount_dataset(dataset)
        if calibration_dataset:
            self.calibration_config = None
            self.calibration_dataset = None
            self.mount_calibration_dataset(calibration_dataset, config.calibration_dataset)

    def mount_dataset(self, dataset: Dataset, analyse: bool = False):
        assert isinstance(dataset, Dataset)

        self.window_size = self.window_size * dataset.sample_rate
        self.step_size = int(self.step_size * dataset.sample_rate)
        self.freeze_time = self.freeze_time * dataset.sample_rate
        self.buffer_size = self.buffer_size * dataset.sample_rate
        self.dataset = shift_data(self.start_time, dataset)

        if analyse:
            self._analyse_dataset()

    def mount_calibration_dataset(self, dataset: Dataset, config='default'):
        assert isinstance(dataset, Dataset)
        self.calibration_dataset = dataset
        get_logger().info('Calibration Dataset mounted.')
        time.sleep(1)

        if config == 'default':
            with open('json_configs/default.json') as c_config:
                self.calibration_config = AttrDict(json.load(c_config))
                get_logger().info('No config was provided for calibration dataset - using default.')
                self._print_config()
        else:
            self.calibration_config = AttrDict(config)
            self._print_config()

    def _print_config(self):
        get_logger().info(f'Start time of dataset: frequency {self.calibration_config.start_time}')
        get_logger().info(f'EMG channel is located at: {self.calibration_config.EMG_Channel}')
        get_logger().info(f'EMG Filtering is performed using butterworth {self.calibration_config.emg_btype}')
        get_logger().info(f'EMG Cutoff is {self.calibration_config.emg_cutoff} Hz')
        get_logger().info(f'EMG Order is {self.calibration_config.emg_order}')
        get_logger().info(f'EEG channels are located at: {self.calibration_config.EEG_Channels}')
        get_logger().info(f'EEG Filtering is performed using butterworth {self.calibration_config.eeg_btype}')
        get_logger().info(f'EEG Cutoff is {self.calibration_config.eeg_cutoff}')
        get_logger().info(f'EEG Order is {self.calibration_config.eeg_order}')
        get_logger().info(f'Window size is {self.calibration_config.window_size} seconds')

    def calibrate(self):
        assert self.calibration_dataset
        assert self.calibration_config

        get_logger().info(f'Shifting Dataset according to config.start_time ({self.calibration_config.start_time})')
        self.calibration_dataset = shift_data(freq=self.calibration_config.start_time, dataset=self.calibration_dataset)

        get_logger().info('Performing MRCP detection on calibration dataset.')

        peaks_to_find = input('Enter the amount of movements expected to find in the dataset. \n')
        windows, scaler = mrcp_detection_for_calibration(data=self.calibration_dataset, input_peaks=int(peaks_to_find),
                                                 config=self.calibration_config)

        self.normalization = scaler

        get_logger().info('MRCP detection complete - displaying average for each channel.')
        average = average_channel(windows)
        plot_average_channels(average, self.calibration_config, layout='grid')

        channel_choice = input('Choose channel to plot for window selecting.\n')
        for window in windows:
            if window.label == 1 and not window.is_sub_window:
                window.plot(channel=int(channel_choice))

        windows = self._select_windows(windows)

        get_logger().info('MRCP Window selection is complete the new average channels are being created.')
        average = average_channel(windows)
        plot_average_channels(average, self.calibration_config, layout='grid')

        get_logger().info('Creating uniform distributed dataset of labels')
        uniform_data = create_uniform_distribution(windows)

        model_selection = self._select_model()

        self._optimize_channels(uniform_data, model_selection)
        get_logger().info('Calibration is complete.')

    def _select_model(self):
        # todo check input is valid (len, availability...)
        while True:
            model_selection = input('Select model knn, svm, lda. \n')

            answer = input(f'Are you sure you want to select {model_selection}? [Y/n]\n')
            if answer.lower() == 'y':
                get_logger().info(f'Selected Model for training and simulation {model_selection}')
                return model_selection

    def _select_windows(self, windows):
        # todo check input is valid (len, availability...)
        while True:

            images_to_remove = str.split(input(
                "Enter MRCP ids to remove windows using format '0 4 5'.. If no windows should be removed enter. \n"))
            if images_to_remove:
                images_to_remove = [int(x) for x in images_to_remove]
            answer = input(f'Deleting {images_to_remove}, Correct? [Y/n]\n')

            if answer.lower() == 'y':
                if images_to_remove:
                    assert isinstance(images_to_remove, list)
                    images_to_remove.sort(reverse=True)
                    for i in range(len(windows) - 1, -1, -1):
                        if i in images_to_remove:
                            del windows[i]
                get_logger().info(f'Deleted {images_to_remove}')
                return windows

    def _optimize_channels(self, data, model):

        self.calibration_score, model, self.EEG_channels = optimize_channels(data, model, self.calibration_config.EEG_Channels)

        get_logger().info(f'The highest optimized score was \n {self.calibration_score}')
        get_logger().info(f'Found using the EEG channels {self.EEG_channels}')
        self.load_models(model)

    def load_models(self, models):
        if isinstance(models, list):
            assert (len(models) == len(self.EEG_channels))
            self.model = models
            get_logger().info(f'Loaded models: {models[0]}')
        else:
            self.model = models
            get_logger().info(f'Loaded model: {models}')

    def simulate(self, real_time: bool, description: bool = True):
        assert bool(self.dataset)

        self.real_time = real_time
        if description:
            self._simulation_information()

        self.iteration = 0
        self.time = time.time()

        self._real_time_check()

        simulation_duration = len(self.dataset.data_device1) - self.step_size
        with tqdm(total=len(self.dataset.data_device1), file=sys.stdout) as pbar:
            while self.iteration < simulation_duration:
                self.sliding_window.frequency_range = [self.iteration - self.window_size, self.iteration]
                if self.freeze_flag:
                    self._freeze_module(pbar)

                elif self.data_buffer_flag:
                    with tqdm(total=self.buffer_size, position=0, file=sys.stdout) as data_buffer_pbar:
                        while len(self.data_buffer) < self.buffer_size:
                            self._build_data_buffer(data_buffer_pbar, pbar)

                        self._initiate_simulation(pbar)
                else:
                    self.sliding_window.data = pd.concat(
                        [self.sliding_window.data.iloc[self.step_size:],
                         self.dataset.data_device1.iloc[self.iteration: self.iteration + self.step_size]],
                        ignore_index=True)

                    self.data_buffer = pd.concat([self.data_buffer.iloc[self.step_size:],
                                                  self.sliding_window.data.iloc[-self.step_size:]],
                                                 ignore_index=True)

                    assert (len(self.data_buffer) == self.buffer_size)
                    assert (len(self.sliding_window.data) == self.window_size)

                    if self.filtering:
                        self._filter_module()
                        self.sliding_window.create_feature_vector()

                        assert (len(self.sliding_window.filtered_data) == len(self.sliding_window.data))
                        # Remove the oldest prediction before making new prediction
                        self.prev_pred_buffer.pop(0)
                        if isinstance(self.model, list) and len(self.EEG_channels) > 1:
                            self.prev_pred_buffer.append(self._prediction_module())
                        else:
                            get_logger().error(f'Single model was provided for multiple channels. ')

                    self._check_heuristic()

                    # evaluate metrics (possibly every x iteration, possibly during freeze time?
                    if self.mrcp_detected:
                        self.freeze_flag = True

                    # Insert sliding windows into sqlite db
                    # self.sliding_window.data.iloc[-self.step_size:, self.EEG_channels].to_sql(name=f'Windows',
                    #                                                                           con=connex,
                    #                                                                           index=False,
                    #                                                                           if_exists='append')

          
                    # update time
                    self._time_module(pbar)

        self._post_simulation_analysis()

    # if metrics are provided, they must follow the convention (target 'arr-like', predictions 'arr-like')
    def evaluation_metrics(self, metrics=None):
        if self.index is None:
            get_logger().warning('No index specified for this dataset, metrics cannot be calculated.')
        else:
            if metrics is None:
                self.metrics = [accuracy, precision, recall, f1]
            else:
                self.metrics = metrics

    def _simulation_information(self):
        get_logger().info('-- # Simulation Description # --')
        get_logger().info(f'Window size: {self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'Initial step size : {self.step_size / self.dataset.sample_rate} seconds')
        get_logger().info(
            f'Buffer size: {self.buffer_size / self.window_size * self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'EEG Channels: {self.EEG_channels}')
        get_logger().info(f'Filtering: {bool(self.filtering)}')
        get_logger().info(f'Freeze Time: {self.freeze_time / self.dataset.sample_rate} seconds.')
        get_logger().info(f'Dataset ID: {self.cue_set}')
        self._dataset_information()
        self._metric_information()

    def _dataset_information(self):
        get_logger().info(
            f'Dataset takes estimated: '
            f'{datetime.timedelta(seconds=round(len(self.dataset.data_device1) / self.dataset.sample_rate))} to simulate.')
        get_logger().info(f'Dataset is sampled at: {self.dataset.sample_rate} frequency.')
        get_logger().info(f'Dataset contains: {(len(self.dataset.TriggerPoint)) // 2} cue onsets.')

    def _build_data_buffer(self, bpbar, pbar):
        self.data_buffer = pd.concat(
            [self.data_buffer, self.dataset.data_device1.iloc[self.iteration:self.iteration + self.step_size]],
            ignore_index=True)
        bpbar.update(self.step_size)
        self._time_module(pbar)

    def _time_module(self, pbar):
        if self.real_time:
            time_after = time.time()
            elapsed_time = time_after - self.time
            sleep_time = (self.allowed_time - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time * TIME_TUNER)
            else:
                get_logger().warning(
                    f'The previous iteration too more than {self.allowed_time} '
                    f'to process - increasing sliding window distance.')
                self.step_size += TIME_PENALTY
                self.allowed_time = self.step_size
            # set time again for next iteration
            self.time = time.time()

        # update iteration and progress bar
        self.iteration += self.step_size
        pbar.update(self.step_size)

    def _filter_module(self):
        filtered_data = pd.DataFrame(columns=self.EEG_channels)

        for channel in self.EEG_channels:
            filtered_data[channel] = butter_filter(data=self.data_buffer[channel],
                                                   order=self.EEG_order,
                                                   cutoff=self.EEG_cutoff,
                                                   btype=self.EEG_btype
                                                   )

        filtered_data[self.EEG_channels] = self.normalization.transform(filtered_data[self.EEG_channels])
        self.sliding_window.filtered_data = filtered_data.iloc[-self.window_size:].reset_index(drop=True)

    def _freeze_module(self, pbar):
        temp_freeze_time = 0
        self.freeze_counter += 1
        self._eval_performance()
        self._apply_metrics()

        while self.freeze_time > temp_freeze_time:
            self.data_buffer = pd.concat(
                [self.data_buffer, self.dataset.data_device1.iloc[self.iteration:self.iteration + self.step_size]],
                ignore_index=True)
            self._time_module(pbar)
            temp_freeze_time += self.step_size

        pbar.set_postfix_str(
            f'{self._dict_score_pp()}')
        self.freeze_flag = False
        self.mrcp_detected = False
        self.prev_pred_buffer = [0] * 5
        self.data_buffer = self.data_buffer[temp_freeze_time:]

    def _dict_score_pp(self):
        a = [{k: round(v, 2) for k, v in self.score.items()}]
        str = ''
        for k, v in a[0].items():
            str += f'{k}: {v} '
        return f'Metrics - {str}'

    def _initiate_simulation(self, pbar):
        self.sliding_window.data = self.data_buffer[-self.window_size:]
        get_logger().info(
            f'Data buffer build of size '
            f'{self.buffer_size / self.window_size * self.window_size / self.dataset.sample_rate} seconds.')
        get_logger().info(f'--- Starting Simulation ---')
        self.data_buffer_flag = False
        self._time_module(pbar)

    def _real_time_check(self):
        if not self.real_time:
            get_logger().info('---------- ########## ----------')
            get_logger().info('Real time simulation disabled.')
        else:
            get_logger().info('---------- ########## ----------')
            get_logger().info('Real time simulation enabled.')
        get_logger().info('Building Data Buffer.')

    def _check_heuristic(self):
        # 4 of the last 5 predictions are MRCP
        if sum(self.prev_pred_buffer) == 4:
            self.mrcp_detected = True

        # maybe also check if dip is y amplitude or something (hard to do because there are 9 channels to check through)

    def _analyse_dataset(self):
        data_buffer_consume = convert_freq_to_datetime(self.buffer_size, self.dataset.sample_rate)
        self._check_cue_skips(data_buffer_consume, is_tp=True)

    def _check_cue_skips(self, time_skip, is_tp):
        skip_rows = []
        if is_tp:
            for i, row in self.dataset.TriggerPoint.iterrows():
                skip_rows.append((row['Date'] < self.dataset.time_start_device1 + time_skip).iloc[0]['Date'])
        if any(skip_rows) and is_tp and sum(skip_rows) > 1:
            get_logger().warning(
                f'The current buffer size will skip the first {(sum(skip_rows) - 1) // 2} TriggerPoints.')

    def _metric_information(self):
        if self.metrics is not None:
            get_logger().info(f'Metrics used for evaluation:')
            for metric in self.metrics:
                get_logger().info(f'{metric.__name__}')

    def _build_index(self):
        raw_index = load_index_list(self.index)
        pair_index = pair_index_list(raw_index)
        self.index = pair_index

        get_logger().info(f'Index loaded for simulation dataset id: {self.cue_set}')
        get_logger().info(f'The index was created on {datetime.datetime.fromtimestamp(self.index_time)}.')

    def _prediction_module(self):
        predictions = []
        for channel in range(0, len(self.EEG_channels)):
            feature_vector = np.array(self.sliding_window.feature_vector[self.EEG_channels[channel]].iloc[0]).flatten()

            predictions.append(self.model[channel].predict([feature_vector]).tolist()[0])

        return collections.Counter(predictions).most_common()[0][0]

    def _eval_performance(self):
        frequency_range = self.sliding_window.frequency_range
        self.predictions.append(1)
        self.prediction_frequency.append(frequency_range)
        is_in_index = False

        for freq in frequency_range:
            for pair in self.index:
                if pair[0] < freq < pair[1]:
                    is_in_index = True

        if is_in_index:
            self.true_labels.append(1)
        else:
            self.true_labels.append(0)

    def _apply_metrics(self):
        for metric in self.metrics:
            self.score[metric.__name__] = metric(self.true_labels, self.predictions)

    def _post_simulation_analysis(self):
        get_logger().info(f' --- Post Simulation Analysis ---')
        get_logger().info(f'{self._dict_score_pp()}')
        self._furthest_prediction_from_mrcp()
        self._distance_to_nearest_mrcp()

    def _distance_to_nearest_mrcp(self):
        distances = []
        for freq_range in self.prediction_frequency:
            min_distance = sys.maxsize
            for i in freq_range:
                for pair in self.index:
                    if pair[0] < i < pair[1]:
                        min_distance = 0
                    else:
                        dist = abs(i - pair[0])
                        if dist < min_distance:
                            min_distance = dist
                        dist = abs(i - pair[1])
                        if dist < min_distance:
                            min_distance = dist
            distances.append(min_distance)
        # iterate over pred freq and closest mrcp pair
        mean_distance = (sum(distances) / len(self.prediction_frequency) / self.dataset.sample_rate)
        mean_missed_distance = (sum([i for i in distances if i != 0]) / len(
            [i for i in self.true_labels if i == 0])) / self.dataset.sample_rate

        get_logger().info(
            f'Prediction lying furthest from MRCP windows {round(max(distances) / self.dataset.sample_rate, 2)} seconds.')
        get_logger().info(f'Mean time of distances from predictions to MRCP window: {round(mean_distance, 2)} seconds.')
        get_logger().info(
            f'Mean time for missed predictions to nearest window: {round(mean_missed_distance, 2)} seconds.')

    def _furthest_prediction_from_mrcp(self):
        found_mrcp = []
        discarded_mrcp = self.buffer_size + self.start_time
        furthest_distance = []

        for pair in self.index:
            if pair[0] < discarded_mrcp or pair[1] < discarded_mrcp:
                found_mrcp.append(999)
                continue
            found = False
            for p in pair:
                for i in self.prediction_frequency:
                    if i[0] < p < i[1]:
                        found = True
            min_distance = sys.maxsize
            if not found:
                for freq in self.prediction_frequency:
                    for i in freq:
                        dist = abs(i - pair[0])
                        if dist < min_distance:
                            min_distance = dist
                        dist = abs(i - pair[1])
                        if dist < min_distance:
                            min_distance = dist
                furthest_distance.append(min_distance)
            found_mrcp.append(found)

        discard_counter = 0
        for i in found_mrcp:
            if i == 999:
                discard_counter += 1

        get_logger().info(f'Total Predictions made by the model {len(self.predictions)}.')
        get_logger().info(f'Total MRCP found in index for dataset: {len(self.index)}.')
        get_logger().info(f'Data buffer removed {discard_counter} MRCP window(s) during building process.')
        get_logger().info(
            f'Correctly predicted {sum(found_mrcp[discard_counter:])}/{len(found_mrcp[discard_counter:])} MRCP Windows.')
        get_logger().info(
            f'The most missed MRCP window had {round(max(furthest_distance) / self.dataset.sample_rate, 2)} seconds '
            f'to the nearest prediction.')
