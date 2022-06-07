import copy
import os
import sys
import uuid
import mne.decoding
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import neurokit2 as nk
import matplotlib.pyplot as plt

from data_training.XGBoost.xgboost_hub import xgboost_training
from data_training.measurements import accuracy
from definitions import OUTPUT_PATH
from utility.logger import get_logger, result_logger
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_training.measurements import accuracy
from utility.logger import get_logger
from tqdm import tqdm
from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter

TIME_PENALTY = 60  # 50 ms
TIME_TUNER = 1  # 0.90  # has to be adjusted to emulate real time properly.
BLINK = 999


class Simulation:

    def __init__(self, config, real_time: bool = False):
        self.time = None
        self.iteration = 0
        self.real_time = real_time
        self.sliding_window = None
        self.model = None
        self.PREV_PRED_SIZE = 0
        self.metrics = None  # maybe implement metrics class
        self.prev_pred_buffer = []
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
        self.extraction_method = None
        self.logger_location = None
        self.dwell_snapshots = []
        self.dwell_true_snapshots = []
        self.dwell_true_buffer = []
        self.INTERNAL_DWELL_FLAG = False
        self.buffer_size = None
        self.dwell = None
        self.data_buffer = pd.DataFrame(columns=config.EEG_CHANNELS)

    def mount_dataset(self, dataset: Dataset):
        assert isinstance(dataset, Dataset)

        self.window_size = int(self.config.window_size * dataset.sample_rate)
        self.step_size = int(self.config.step_size * dataset.sample_rate)
        self.buffer_size = int(self.config.buffer_size * dataset.sample_rate)
        self.PREV_PRED_SIZE = int((self.window_size / self.step_size) / 2)  # half the window size
        self.prev_pred_buffer = [0] * self.PREV_PRED_SIZE
        self.dwell_true_buffer = [0] * self.PREV_PRED_SIZE
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

    def set_feature_extraction(self, extract, method):
        if extract:
            self.feature_extraction = True
            self.extraction_method = method
        else:
            self.feature_extraction = False

    def set_logger_location(self, path):
        self.logger_location = path

    def simulate(self, real_time: bool, description: bool = True, analyse: bool = True):
        assert bool(self.dataset)

        self.real_time = real_time
        if description:
            self._simulation_information()
            self._real_time_check()

        blinks = self._blink_detection()

        self.iteration = 0

        time.sleep(1)

        simulation_duration = len(self.dataset.data) - self.step_size
        with tqdm(total=len(self.dataset.data), file=sys.stdout) as pbar:
            while self.iteration < simulation_duration:
                # start, middle, end
                self.frequency_range = [self.iteration,
                                        self.iteration + self.window_size / 2,
                                        self.iteration + self.window_size]
                if self.freeze_flag:
                    self._freeze_module(pbar)

                elif self.data_buffer_flag:
                    self.time = time.time()

                    with tqdm(total=self.buffer_size, position=0, file=sys.stdout) as data_buffer_pbar:
                        while len(self.data_buffer) < self.buffer_size:
                            self._build_data_buffer(data_buffer_pbar, pbar)

                    self._initiate_simulation(pbar)
                else:
                    self.data_buffer = pd.concat([self.data_buffer.iloc[
                                                  self.step_size:
                                                  ],
                                                  self.dataset.data.iloc[
                                                  self.iteration:
                                                  self.iteration + self.step_size]
                                                  ],
                                                 ignore_index=True)

                    if not (len(self.data_buffer) == self.buffer_size):
                        get_logger().error('something went wrong with the databuffer')
                        get_logger().error(f'len data buffer {len(self.data_buffer)}')

                    if self.filter:
                        self._filter_module(self.filter)

                    # Remove the oldest prediction before making new prediction
                    if len(self.prev_pred_buffer) >= self.PREV_PRED_SIZE:
                        self.prev_pred_buffer.pop(0)

                        if self.INTERNAL_DWELL_FLAG:
                            self.dwell_true_buffer.pop(0)

                    # Check if a blink is in the moving window
                    skip_prediction = False
                    for b in blinks:
                        if self.frequency_range[0] < b < self.frequency_range[-1]:
                            skip_prediction = True
                            self.prev_pred_buffer.append(BLINK)

                            if self.INTERNAL_DWELL_FLAG:
                                # //self.dwell_snapshots.append(copy.deepcopy(self.prev_pred_buffer))
                                self.dwell_true_buffer.append(BLINK)
                                # self.dwell_true_snapshots.append(copy.deepcopy(self.dwell_true_buffer))
                            break

                    if not skip_prediction:
                        if BLINK in self.prev_pred_buffer:
                            self.prev_pred_buffer = [0] * (self.PREV_PRED_SIZE - 1)
                            if self.INTERNAL_DWELL_FLAG:
                                self.dwell_true_buffer = [0] * (self.PREV_PRED_SIZE - 1)

                        self.prev_pred_buffer.append(self._prediction_module())

                        if self.INTERNAL_DWELL_FLAG:
                            self.dwell_snapshots.append(copy.deepcopy(self.prev_pred_buffer))
                            t = False
                            for cluster in self.dataset.clusters:
                                if self.frequency_range[0] < cluster.start < self.frequency_range[2]:
                                    t = True
                                    self.dwell_true_buffer.append(1)
                                    break
                            if not t:
                                self.dwell_true_buffer.append(0)
                            self.dwell_true_snapshots.append(copy.deepcopy(self.dwell_true_buffer))

                        if not self.INTERNAL_DWELL_FLAG:
                            # if running dwell we don't want to freeze the system
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
        get_logger().info('---------- Dwell Tuning\n')
        # runs the dwell dataset and should trigger the least amount of times.
        # run simulation on the dwell dataset
        self.INTERNAL_DWELL_FLAG = True
        self.reset()
        self.mount_dataset(dwell_dataset)
        self.simulate(real_time=False, description=False, analyse=False)

        v0 = [sum(y) for y in self.dwell_true_snapshots]
        v1 = [sum(x) for x in self.dwell_snapshots]

        indexes = [i for i, v in enumerate(v0) if v == 10]

        comp0 = []
        comp1 = []
        for ix in indexes:
            comp0.append(v0[ix])
            comp1.append(v1[ix])

        number_of_peaks = []
        var = []
        cluster_start = False
        i = 0
        while i < len(v0) - 1:
            if v0[i + 1] > 0 and not cluster_start:
                cluster_start = True
                var.append(i + 1)
            if cluster_start and v0[i + 1] == 0:
                cluster_start = False
                var.append(i)
                number_of_peaks.append(var)
                var = []
            i += 1

        comp1.sort()
        median = comp1[int(len(comp1) / 2)]
        # upper_percentile = comp1[int(len(comp1) * 0.9)]

        def evaluate_dwell(dwell):
            correct_clusters_found = 0
            correct_triggers = 0
            INITIALIZED = False
            while correct_triggers == correct_clusters_found or not INITIALIZED:
                indexes_where_dwell_triggers = [i for i, v in enumerate(v1) if v >= dwell]
                res = [1 if v0[x] > 0 else 0 for x in indexes_where_dwell_triggers]

                # tp = sum(res) / len(res)

                correct_triggers = 0
                for (s, e) in number_of_peaks:
                    cluster_met = False
                    for i in indexes_where_dwell_triggers:
                        if s < i < e:
                            cluster_met = True
                    if cluster_met:
                        correct_triggers += 1

                counter = 0
                for i in indexes_where_dwell_triggers:
                    in_a_cluster = False
                    for (s, e) in number_of_peaks:
                        if s < i < e:
                            in_a_cluster = True
                    if in_a_cluster:
                        counter += 1

                get_logger().info(
                    f'Dwell = {dwell}, Clusters Hit = {correct_triggers}/{len(number_of_peaks)}, Predictions in a cluster {counter} / {len(indexes_where_dwell_triggers)} ({round(counter / len(indexes_where_dwell_triggers), 2)})')
                result_logger(self.logger_location,
                              f'Dwell = {dwell}, Clusters Hit = {correct_triggers}/{len(number_of_peaks)}, Predictions in a cluster {counter} / {len(indexes_where_dwell_triggers)} ({round(counter / len(indexes_where_dwell_triggers), 2)}) \n')

                if dwell == 10:
                    return dwell
                if not INITIALIZED:
                    correct_clusters_found = correct_triggers
                    INITIALIZED = True
                if correct_triggers != correct_clusters_found:
                    return dwell-1
                else:
                    dwell += 1

        # first dwell dataset set the initial dwell
        if self.dwell is None:
            self.dwell = evaluate_dwell(median)
        # 2nd dwell dataset take the average of two dwells
        else:
            self.dwell = (self.dwell + evaluate_dwell(median)) // 2
        self.reset()
        self.INTERNAL_DWELL_FLAG = False

        #get_logger().info(f'Acceptable level reached of {self.freeze_counter}'
        #                  f' false positive predictions in a '
        #                  f'{datetime.timedelta(seconds=round(len(dwell_dataset.data) / self.dataset.sample_rate))}.')
        #get_logger().info(f'Dwell parameter adjusted to be {len(self.prev_pred_buffer)} consecutive predictions.')

        # if self.logger_location is not None:
        #     result_logger(self.logger_location, f'Dwell Tuning -- \n')
        #     result_logger(self.logger_location, f'Acceptable level reached of {self.freeze_counter}'
        #                                        f' false positive predictions in a '
        #                                        f'{datetime.timedelta(seconds=round(len(dwell_dataset.data) / self.dataset.sample_rate))} session\n')
        #    result_logger(self.logger_location,
        #                  f'Dwell parameter adjusted to be {len(self.prev_pred_buffer)} consecutive predictions.\n')

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
        self.dwell_snapshots = []
        self.dwell_true_snapshots = []
        self.dwell_true_buffer = []
        self.score = {}

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
        if sum(self.prev_pred_buffer) == self.dwell:
            self.freeze_flag = True

    def _metric_information(self):
        if self.metrics is not None:
            get_logger().info(f'Metrics used for evaluation:')
            for metric in self.metrics:
                get_logger().info(f'{metric.__name__}')

    def _prediction_module(self):
        if self.feature_extraction:
            if isinstance(self.extraction_method, mne.decoding.CSP):
                reshaped = self.sliding_window.reshape(1, self.sliding_window.shape[1], self.sliding_window.shape[0])
                features = self.extraction_method.transform(reshaped)
            else:
                features = self.extraction_method([self.sliding_window])
            return self.model.predict(features)[0]
        else:
            reshaped = self.sliding_window.reshape(1, self.sliding_window.shape[1], self.sliding_window.shape[0], 1)
            prediction = self.model.predict(reshaped)[0][0]
            # return closest int
            return round(prediction)

    def _eval_performance(self):
        self.predictions.append(1)
        self.prediction_frequency.append(self.frequency_range)
        is_in_index = False

        # we only evaluate center of the prediction in inside the intended range
        for cluster in self.dataset.clusters:
            if self.frequency_range[0] < cluster.start < self.frequency_range[2]:
                self.true_labels.append(1)
                # return the end of the cluster
                return max(cluster.end, self.iteration + 2 * self.dataset.sample_rate)
        if not is_in_index:
            self.true_labels.append(0)
            # break windowlength
            return self.iteration + self.config.window_size * self.dataset.sample_rate

    def _apply_metrics(self):
        for metric in self.metrics:
            self.score[metric.__name__] = metric(self.true_labels, self.predictions)

    def _post_simulation_analysis(self):
        get_logger().info(f'---------- Post Simulation Analysis')
        get_logger().info(f'{self._dict_score_pp()}')
        discard_counter, found_mrcp, furthest_distance = self._furthest_prediction_from_mrcp()
        mean_distance, mean_missed_distance, furthest_prediction = self._distance_to_nearest_mrcp()

        self._plot_predictions()

        if self.logger_location is not None:
            result_logger(self.logger_location, 'Simulation Analysis -- \n')
            result_logger(self.logger_location, f'{self._dict_score_pp()} \n')
            result_logger(self.logger_location, f'Total Predictions made by the model {len(self.predictions)}.\n')
            result_logger(self.logger_location,
                          f'Total movement intentions found in index for dataset: {len(self.dataset.clusters)}.\n')
            result_logger(self.logger_location,
                          f'Data buffer removed {discard_counter} movement intention windows during building process.\n')
            result_logger(self.logger_location,
                          f'Correctly predicted {sum(found_mrcp[discard_counter:])}/{len(found_mrcp[discard_counter:])} movement intention windows.\n')
            if len(furthest_distance) > 0:
                result_logger(self.logger_location,
                              f'The most missed intention window had {round(max(furthest_distance) / self.dataset.sample_rate, 2)} seconds to the nearest prediction.\n')
            else:
                result_logger(self.logger_location, 'All movement intention windows were hit!\n')
            result_logger(self.logger_location,
                          f'Prediction lying furthest from intention windows {furthest_prediction} seconds.\n')
            result_logger(self.logger_location,
                          f'Mean time of distances from predictions to intention window: {mean_distance} seconds.\n')
            result_logger(self.logger_location,
                          f'Mean time for missed predictions to nearest window: {mean_missed_distance} seconds.\n')

    def _distance_to_nearest_mrcp(self):
        # note: freq_range[1] refers to the center of the prediction window
        distances = []
        for freq_range in self.prediction_frequency:
            min_distance = sys.maxsize
            for cluster in self.dataset.clusters:
                if freq_range[0] < cluster.start < freq_range[1]:
                    min_distance = 0
                else:
                    dist = abs(freq_range[0] - cluster.start)
                    if dist < min_distance:
                        min_distance = dist
                    dist = abs(freq_range[-1] - cluster.start)
                    if dist < min_distance:
                        min_distance = dist
            distances.append(min_distance)
        # iterate over pred freq and closest mrcp pair
        mean_distance = (sum(distances) / len(self.prediction_frequency) / self.dataset.sample_rate)
        mean_missed_distance = (sum([i for i in distances if i != 0]) / len(
            [i for i in self.true_labels if i == 0])) / self.dataset.sample_rate

        return round(mean_distance, 2), round(mean_missed_distance, 2), \
               round(max(distances) / self.dataset.sample_rate, 2)

    def _furthest_prediction_from_mrcp(self):
        found_mrcp = []
        discarded_mrcp = self.buffer_size
        furthest_distance = []

        for cluster in self.dataset.clusters:
            if cluster.start - self.dataset.sample_rate * 2 < discarded_mrcp or \
                    cluster.start + self.dataset.sample_rate < discarded_mrcp:
                found_mrcp.append(999)
                continue
            found = False
            for prediction_freq in self.prediction_frequency:
                if prediction_freq[0] < cluster.start < prediction_freq[-1]:
                    found = True
            min_distance = sys.maxsize
            if not found:
                for prediction_freq in self.prediction_frequency:
                    dist = abs(prediction_freq[0] - cluster.start)
                    if dist < min_distance:
                        min_distance = dist
                    dist = abs(prediction_freq[-1] - cluster.start)
                    if dist < min_distance:
                        min_distance = dist
                furthest_distance.append(min_distance)
            found_mrcp.append(found)

        discard_counter = 0
        for i in found_mrcp:
            if i == 999:
                discard_counter += 1

        return discard_counter, found_mrcp, furthest_distance

    def _plot_predictions(self):
        plt.clf()
        fig = plt.figure(figsize=(40, 8))
        plot_arr = []
        max_height = self.dataset.filtered_data.max()
        blinks = self._blink_detection()

        for cluster in self.dataset.clusters:
            # dashed lines 2 seconds before onset and 1 second after

            # plt.vlines(cluster.start - self.dataset.sample_rate * 2, 0, max_height, linestyles='--', color='black')
            plt.vlines(cluster.start, 0, max_height, linestyles='--', color='black')
            plt.axvspan(cluster.start - self.dataset.sample_rate * self.config.window_size,
                        cluster.start + self.dataset.sample_rate,
                        alpha=0.80,
                        color='lightblue')
            plot_arr.append(self.dataset.filtered_data.iloc[cluster.start:cluster.end])

        plt.plot(np.abs(self.dataset.filtered_data), color='black')
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
        for b in blinks:
            size = self.prediction_frequency[0]
            #
            plt.gca().add_patch(
                patches.Rectangle((b, max_height * 0.45), abs(size[0] - size[-1]) * 0.3,
                                  max_height * 0.025, linewidth=1, alpha=1, fill=False, edgecolor='black')
            )
        plt.axvspan(0, self.config.buffer_size * self.dataset.sample_rate, alpha=0.80, color='yellow')

        plt.xlabel('Time (s)')
        plt.ylabel('mV (Filtered)', labelpad=-2)
        plt.autoscale()

        if self.logger_location is not None:
            # save_figure(os.path.join(OUTPUT_PATH, 'results', self.logger_location[:-4]), fig)
            fig.savefig(os.path.join(OUTPUT_PATH, 'results', self.logger_location[:-4] + str(uuid.uuid4())))

        # plt.show()

    def _blink_detection(self):
        eog_cleaned = nk.eog_clean(self.dataset.data[self.config.EOG_CHANNEL], sampling_rate=self.dataset.sample_rate,
                                   method='agarwal2019')
        blinks = nk.eog_findpeaks(eog_cleaned, sampling_rate=self.dataset.sample_rate, method='neurokit', show=False)

        return blinks
