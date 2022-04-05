import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from classes.Dataset import Dataset
import time
import matplotlib.pyplot as plt
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_training.measurements import accuracy
from utility.logger import get_logger
import datetime
from data_preprocessing.filters import butter_filter
import matplotlib.patches as patches

TIME_PENALTY = 60  # 50 ms
TIME_TUNER = 1  # 0.90  # has to be adjusted to emulate real time properly.


class Simulation:

    def __init__(self, config, real_time: bool = False):
        self.time = None
        self.iteration = 0
        self.real_time = real_time
        self.sliding_window = None
        self.model = None
        self.PREV_PRED_SIZE = 1
        self.metrics = None  # maybe implement metrics class
        self.prev_pred_buffer = [0] * self.PREV_PRED_SIZE
        self.freeze_flag = False
        self.freeze_counter = 0
        self.data_buffer_flag = True
        self.predictions = []
        self.prediction_frequency = []
        self.true_labels = []
        self.score = {}
        self.normalization = None
        self.frequency_range = []
        self.window_size = None
        self.step_size = None
        self.dataset = None
        self.config = config
        self.filter = None
        self.feature_extraction = None
        self.buffer_size = self.config.window_size * config.buffer_size
        self.data_buffer = pd.DataFrame(columns=config.EEG_CHANNELS)

    def mount_dataset(self, dataset: Dataset):
        assert isinstance(dataset, Dataset)

        self.window_size = int(self.config.window_size * dataset.sample_rate)
        self.step_size = int(self.config.step_size * dataset.sample_rate)
        self.buffer_size = int(self.config.buffer_size * dataset.sample_rate)
        self.dataset = dataset

    def load_models(self, models):
        if isinstance(models, list):
            assert (len(models) == len(self.config.EEG_channels))
            self.model = models
            get_logger().info(f'Loaded models: {models[0]}')
        else:
            self.model = models
            get_logger().info(f'Loaded model: {models}')

    def set_normalizer(self, normalization):
        self.normalization = normalization

    def set_filter(self, filter_range):
        self.filter = filter_range

    def set_feature_extraction(self, extract):
        if extract:
            self.feature_extraction = True
        else:
            self.feature_extraction = False

    def simulate(self, real_time: bool, description: bool = True, analyse: bool = True):
        assert bool(self.dataset)

        self.real_time = real_time
        if description:
            self._simulation_information()
            self._real_time_check()

        self.iteration = 0

        time.sleep(1)

        simulation_duration = len(self.dataset.data) - self.step_size
        with tqdm(total=len(self.dataset.data), file=sys.stdout) as pbar:
            while self.iteration < simulation_duration:
                self.frequency_range = [self.iteration, self.iteration + self.window_size]
                if self.freeze_flag:
                    self._freeze_module(pbar)

                elif self.data_buffer_flag:
                    self.time = time.time()

                    with tqdm(total=self.buffer_size, position=0, file=sys.stdout) as data_buffer_pbar:
                        while len(self.data_buffer) < self.buffer_size:
                            self._build_data_buffer(data_buffer_pbar, pbar)

                    self._initiate_simulation(pbar)
                else:
                    self.data_buffer = pd.concat([self.data_buffer.iloc[self.step_size:],
                                                  self.dataset.data.iloc[
                                                  self.iteration:
                                                  self.iteration + self.step_size]
                                                  ],
                                                 ignore_index=True)

                    if not (len(self.data_buffer) == self.buffer_size):
                        print('something went wrong with the databuffer')
                        print(f'len data buffer {(len(self.data_buffer))}')

                    if self.filter:
                        self._filter_module(self.filter)

                    # Remove the oldest prediction before making new prediction
                    self.prev_pred_buffer.pop(0)

                    self.prev_pred_buffer.append(self._prediction_module())

                    self._check_heuristic()

                    # update time
                    self._time_module(pbar)

        if analyse:
            self._post_simulation_analysis()

    # if metrics are provided, they must follow the convention (target 'arr-like', predictions 'arr-like')
    def set_evaluation_metrics(self, metrics=None):
        if metrics is None:
            self.metrics = [accuracy]
        else:
            self.metrics = metrics

    def _simulation_information(self):
        get_logger().info('---------- Simulation Description')
        get_logger().info(f'Window size: {self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'Initial step size : {self.step_size / self.dataset.sample_rate} seconds')
        get_logger().info(
            f'Buffer size: {self.buffer_size / self.window_size * self.window_size / self.dataset.sample_rate} seconds')
        self._dataset_information()
        self._metric_information()

    def _dataset_information(self):
        get_logger().info(
            f'Dataset takes estimated: '
            f'{datetime.timedelta(seconds=round(len(self.dataset.data) / self.dataset.sample_rate))} to '
            f'simulate.')
        get_logger().info(f'Dataset is sampled at: {self.dataset.sample_rate} frequency.')
        # get_logger().info(f'Dataset contains: {(len(self.dataset.TriggerPoint)) // 2} cue onsets.')

    def _build_data_buffer(self, bpbar, pbar):
        self.data_buffer = pd.concat([
            self.data_buffer,
            self.dataset.data.iloc[self.iteration:
                                   self.iteration + self.step_size]
        ],
            ignore_index=True)
        bpbar.update(self.step_size)
        self._time_module(pbar)

    def tune_dwell(self, dwell_dataset):
        get_logger().info('---------- Dwell Tuning')
        # runs the dwell dataset and should trigger the least amount of times.
        # run simulation on the dwell dataset
        self.freeze_counter = 6  # just to start the first iteration

        while self.freeze_counter > 4:
            self.PREV_PRED_SIZE += 1
            self.reset()
            self.mount_dataset(dwell_dataset)
            self.simulate(real_time=False, description=False, analyse=False)
            get_logger().info(f'Dwell = {self.PREV_PRED_SIZE}, FP = {self.freeze_counter}')

        get_logger().info(f'Acceptable level reached of {self.freeze_counter}'
                          f' false positive predictions in a '
                          f'{datetime.timedelta(seconds=round(len(dwell_dataset.data) / self.dataset.sample_rate))}.')
        get_logger().info(f'Dwell parameter adjusted to be {len(self.prev_pred_buffer)} consecutive predictions.')

    def reset(self):
        # reset dataset specific things.
        get_logger().info('---------- Resetting Simulation')
        get_logger().info('Resetting simulation specific data. ')
        self.prev_pred_buffer = [0] * self.PREV_PRED_SIZE
        self.freeze_flag = False
        self.freeze_counter = 0
        self.data_buffer_flag = True
        self.predictions = []
        self.prediction_frequency = []
        self.frequency_range = []
        self.window_size = None
        self.step_size = None
        self.sliding_window = None
        self.data_buffer = pd.DataFrame(columns=self.config.EEG_CHANNELS)
        self.true_labels = []

    def _time_module(self, pbar):
        if self.real_time:
            time_after = time.time()
            elapsed_time = time_after - self.time
            sleep_time = (self.step_size / self.dataset.sample_rate - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time * TIME_TUNER)
            else:
                get_logger().warning(
                    f'The previous iteration too more than {self.step_size} '
                    f'to process - increasing sliding window distance.')
                self.step_size += TIME_PENALTY
            # set time again for next iteration
            self.time = time.time()

        # update iteration and progress bar
        self.iteration += self.step_size
        pbar.update(self.step_size)

    def _filter_module(self, filter_range):
        filtered_data = pd.DataFrame(columns=self.config.EEG_CHANNELS)

        for channel in self.config.EEG_CHANNELS:
            filtered_data[channel] = butter_filter(data=self.data_buffer[channel],
                                                   order=self.config.EEG_ORDER,
                                                   cutoff=filter_range,
                                                   btype=self.config.EEG_BTYPE
                                                   )

        filtered_data[self.config.EEG_CHANNELS] = self.normalization.transform(filtered_data[self.config.EEG_CHANNELS])

        self.sliding_window = np.array(filtered_data.iloc[-self.window_size:].reset_index(drop=True))

    def _freeze_module(self, pbar):
        temp_freeze_time = self.iteration
        self.freeze_counter += 1
        freeze_time = self._eval_performance()
        self._apply_metrics()

        while freeze_time > temp_freeze_time:
            self.data_buffer = pd.concat(
                [self.data_buffer, self.dataset.data.iloc[self.iteration:
                                                          self.iteration + self.step_size]],
                ignore_index=True)
            self._time_module(pbar)
            temp_freeze_time += self.step_size

        pbar.set_postfix_str(
            f'{self._dict_score_pp()}')
        self.freeze_flag = False
        self.prev_pred_buffer = [0] * self.PREV_PRED_SIZE
        self.data_buffer = self.data_buffer[-self.buffer_size:]

    def _dict_score_pp(self):
        a = [{k: round(v, 2) for k, v in self.score.items()}]
        string = ''
        for k, v in a[0].items():
            string += f'{k}: {v} '
        return f'Metrics - {string}'

    def _initiate_simulation(self, pbar):
        self.sliding_window = self.data_buffer[-self.window_size:]
        get_logger().info(
            f'Data buffer build of size '
            f'{self.buffer_size / self.window_size * self.window_size / self.dataset.sample_rate} seconds.')
        get_logger().info(f'---------- Starting Simulation')
        self.data_buffer_flag = False
        self._time_module(pbar)

    def _real_time_check(self):
        if not self.real_time:
            get_logger().info('---------- Real Time')
            get_logger().info('Real time simulation disabled.')
        else:
            get_logger().info('---------- Real Time')
            get_logger().info('Real time simulation enabled.')
        get_logger().info('Building Data Buffer.')

    def _check_heuristic(self):
        if sum(self.prev_pred_buffer) == self.PREV_PRED_SIZE:
            self.freeze_flag = True

    def _metric_information(self):
        if self.metrics is not None:
            get_logger().info(f'Metrics used for evaluation:')
            for metric in self.metrics:
                get_logger().info(f'{metric.__name__}')

    def _prediction_module(self):
        if self.feature_extraction:
            features = extract_features([self.sliding_window])
            return self.model.predict(features)[0]
        else:
            return self.model.predict(self.sliding_window)

    def _eval_performance(self):
        self.predictions.append(1)
        self.prediction_frequency.append(self.frequency_range)
        is_in_index = False

        for freq in self.frequency_range:
            for pair in self.dataset.onsets_index:
                if pair[0] - self.dataset.sample_rate / 2 < freq < pair[0] + self.dataset.sample_rate / 2:
                    self.true_labels.append(1)
                    # return the end of the cluster
                    return pair[2]
        if not is_in_index:
            self.true_labels.append(0)
            return self.iteration + 2 * self.dataset.sample_rate

    def _apply_metrics(self):
        for metric in self.metrics:
            self.score[metric.__name__] = metric(self.true_labels, self.predictions)

    def _post_simulation_analysis(self):
        get_logger().info(f'---------- Post Simulation Analysis')
        get_logger().info(f'{self._dict_score_pp()}')
        self._furthest_prediction_from_mrcp()
        self._distance_to_nearest_mrcp()
        self._plot_predictions()

    def _distance_to_nearest_mrcp(self):
        distances = []
        for freq_range in self.prediction_frequency:
            min_distance = sys.maxsize
            for i in freq_range:
                for pair in self.dataset.onsets_index:
                    if pair[0] - self.dataset.sample_rate / 2 < i < pair[0] + self.dataset.sample_rate / 2:
                        min_distance = 0
                    else:
                        dist = abs(i - pair[0] - self.dataset.sample_rate / 2)
                        if dist < min_distance:
                            min_distance = dist
                        dist = abs(i - pair[0] + self.dataset.sample_rate / 2)
                        if dist < min_distance:
                            min_distance = dist
            distances.append(min_distance)
        # iterate over pred freq and closest mrcp pair
        mean_distance = (sum(distances) / len(self.prediction_frequency) / self.dataset.sample_rate)
        mean_missed_distance = (sum([i for i in distances if i != 0]) / len(
            [i for i in self.true_labels if i == 0])) / self.dataset.sample_rate

        get_logger().info(
            f'Prediction lying furthest from intention windows {round(max(distances) / self.dataset.sample_rate, 2)} seconds.')
        get_logger().info(
            f'Mean time of distances from predictions to intention window: {round(mean_distance, 2)} seconds.')
        get_logger().info(
            f'Mean time for missed predictions to nearest window: {round(mean_missed_distance, 2)} seconds.')

    def _furthest_prediction_from_mrcp(self):
        found_mrcp = []
        discarded_mrcp = self.buffer_size
        furthest_distance = []

        for pair in self.dataset.onsets_index:
            if pair[0] - self.dataset.sample_rate / 2 < discarded_mrcp or \
                    pair[0] + self.dataset.sample_rate / 2 < discarded_mrcp:
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
                        dist = abs(i - pair[0] - self.dataset.sample_rate / 2)
                        if dist < min_distance:
                            min_distance = dist
                        dist = abs(i - pair[0] + self.dataset.sample_rate / 2)
                        if dist < min_distance:
                            min_distance = dist
                furthest_distance.append(min_distance)
            found_mrcp.append(found)

        discard_counter = 0
        for i in found_mrcp:
            if i == 999:
                discard_counter += 1

        get_logger().info(f'Total Predictions made by the model {len(self.predictions)}.')
        get_logger().info(f'Total movement intentions found in index for dataset: {len(self.dataset.onsets_index)}.')
        get_logger().info(f'Data buffer removed {discard_counter} movement intention windows during building process.')
        get_logger().info(
            f'Correctly predicted {sum(found_mrcp[discard_counter:])}/{len(found_mrcp[discard_counter:])} '
            f'movement intention windows.')
        get_logger().info(
            f'The most missed intention window had {round(max(furthest_distance) / self.dataset.sample_rate, 2)}'
            f'seconds to the nearest prediction.')

    def _plot_predictions(self):
        plt.clf()
        plot_arr = []
        max_height = self.dataset.filtered_data[self.config.EMG_CHANNEL].max()

        for cluster in self.dataset.onsets_index:
            plt.vlines(cluster[0]-self.dataset.sample_rate/2, 0, max_height)
            plt.vlines(cluster[0]+self.dataset.sample_rate/2, 0, max_height)
            plot_arr.append(self.dataset.filtered_data[self.config.EMG_CHANNEL].iloc[cluster[0]:cluster[-1]])

        plt.plot(np.abs(self.dataset.filtered_data[self.config.EMG_CHANNEL]), color='black')
        for vals in plot_arr:
            plt.plot(np.abs(vals))

        for prediction, correct in zip(self.prediction_frequency, self.true_labels):
            if correct:
                plt.gca().add_patch(
                    patches.Rectangle((prediction[0], max_height * 0.7), abs(prediction[0] - prediction[-1]),
                                      max_height * 0.1, linewidth=0.5, alpha=0.5, facecolor='green', fill=True,
                                      edgecolor='black'))
            elif not correct:
                plt.gca().add_patch(
                    patches.Rectangle((prediction[0], max_height * 0.4), abs(prediction[0] - prediction[-1]),
                                      max_height * 0.1, linewidth=0.5, alpha=0.5, facecolor='red', fill=True,
                                      edgecolor='black'))

        plt.xlabel('Time (s)')
        plt.ylabel('mV (Filtered)', labelpad=-2)
        plt.autoscale()
        plt.show()
