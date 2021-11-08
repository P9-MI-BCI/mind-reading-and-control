from tqdm import tqdm

from classes.Dataset import Dataset
from classes.Window import Window
import time
import pandas as pd

from data_preprocessing.data_shift import shift_data
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from data_training.measurements import get_accuracy, get_precision, get_recall, get_f1_score
from utility.logger import get_logger
from data_preprocessing.filters import butter_filter
import datetime
import random

TIME_PENALTY = 60  # 50 ms
TIME_TUNER = 1  # 0.90  # has to be adjusted to emulate real time properly.


class Simulation:

    def __init__(self, config, dataset: Dataset = 0, real_time: bool = False):
        self.time = time.time()
        self.iteration = 0
        self.real_time = real_time
        self.sliding_window = Window()
        self.model = None
        self.index = None
        self.metrics = None  # maybe implement metrics class
        self.prev_pred_buffer = [0] * 5
        self.mrcp_detected = False
        self.freeze_flag = False
        self.data_buffer_flag = True

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

        if dataset:
            self.dataset = dataset
            self.mount_dataset(dataset)

    def mount_dataset(self, dataset: Dataset, analyse: bool = False):
        assert isinstance(dataset, Dataset)

        self.window_size = self.window_size * dataset.sample_rate
        self.step_size = int(self.step_size * dataset.sample_rate)
        self.freeze_time = self.freeze_time * dataset.sample_rate
        self.buffer_size = self.buffer_size * dataset.sample_rate
        self.dataset = shift_data(self.start_time, dataset)

        if analyse:
            self._analyse_dataset()

    def load_models(self, models, data_config):
        pass
        # todo when loading models, load appropriate index list
        # index = load_index_list()
        # pair_indexes = pair_index_list(index)

    def simulate(self, real_time: bool, description: bool = True):
        assert bool(self.dataset)

        self.real_time = real_time
        if description:
            self._simulation_information()

        self.iteration = 0
        self.time = time.time()

        self._real_time_check()

        simulation_duration = len(self.dataset.data_device1) - self.step_size
        with tqdm(total=len(self.dataset.data_device1)) as pbar:
            while self.iteration < simulation_duration:
                if self.freeze_flag:
                    self._freeze_module(pbar)

                elif self.data_buffer_flag:
                    with tqdm(total=self.buffer_size, position=0) as data_buffer_pbar:
                        while len(self.data_buffer) < self.buffer_size:
                            self._build_data_buffer(data_buffer_pbar, pbar)

                        self._initiate_simulation(pbar)
                else:
                    self.sliding_window.data = pd.concat(
                        [self.sliding_window.data.iloc[self.iteration:],
                         self.dataset.data_device1.iloc[self.iteration: self.iteration + self.step_size]],
                        ignore_index=True)

                    self.data_buffer = pd.concat([self.data_buffer, self.sliding_window.data[-self.step_size:]],
                                                 ignore_index=True)

                    if self.filtering:
                        self._filter_module()
                        self._feature_extraction_predictions()
                        # temp for testing
                        self.prev_pred_buffer.pop(0)
                        self.prev_pred_buffer.append(random.randint(0, 1))

                    self._check_heuristic()

                    # evaluate metrics (possibly every x iteration, possibly during freeze time?
                    if self.mrcp_detected:
                        self.freeze_flag = True

                    # deallocate data
                    self.data_buffer = self.data_buffer.iloc[self.step_size:]
                    self.sliding_window.data = self.sliding_window.data.iloc[self.step_size:]

                    # update time
                    self._time_module(pbar)

    # if metrics are provided, they must follow the convention (target 'arr-like', predictions 'arr-like')
    def evaluation_metrics(self, metrics=None, concurrently: bool = False):
        if metrics is None:
            self.metrics = [get_accuracy, get_precision, get_recall, get_f1_score]
        else:
            self.metrics = metrics
        if concurrently:
            get_logger().warning('Evaluating metrics concurrently can be detrimental to performance.')

    def _simulation_information(self):
        get_logger().info('-- # Simulation Description # --')
        get_logger().info(f'Window size: {self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'Initial step size : {self.step_size / self.dataset.sample_rate} seconds')
        get_logger().info(
            f'Buffer size: {self.buffer_size / self.window_size * self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'EEG Channels: {self.EEG_channels}')
        get_logger().info(f'Filtering: {bool(self.filtering)}')
        get_logger().info(f'Freeze Time: {self.freeze_time / self.dataset.sample_rate}')
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
            # set time again for next iteration
            self.time = time.time()

        # update iteration and progress bar
        self.iteration += self.step_size
        pbar.update(self.step_size)

    def _feature_extraction_predictions(self):
        for channel in self.EEG_channels:
            feature_vector = []
            for feature in self.sliding_window.get_features():
                f = getattr(self.sliding_window, feature)
                feature_vector.append(f[channel].item())

        # predictions.append(models[channel].predict([feature_vector]).tolist()[0])

    def _filter_module(self):
        filtered_data = pd.DataFrame(columns=self.EEG_channels)

        for channel in self.EEG_channels:
            filtered_data[channel] = butter_filter(data=self.data_buffer[channel],
                                                   order=self.EEG_order,
                                                   cutoff=self.EEG_cutoff,
                                                   btype=self.EEG_btype
                                                   )

        self.sliding_window.filtered_data = filtered_data.iloc[-self.window_size:].reset_index(drop=True)

        self.sliding_window.extract_features()

    def _freeze_module(self, pbar):
        get_logger().info(
            f'MRCP Detected - Freeze components for {self.freeze_time / self.dataset.sample_rate} seconds')
        temp_freeze_time = 0
        while self.freeze_time > temp_freeze_time:
            self.data_buffer = pd.concat(
                [self.data_buffer, self.dataset.data_device1.iloc[self.iteration:self.iteration + self.step_size]],
                ignore_index=True)
            self._time_module(pbar)
            temp_freeze_time += self.step_size

        get_logger().info(
            f'Freeze time ended, functionality resumed'
        )
        self.freeze_flag = False
        self.mrcp_detected = False
        self.prev_pred_buffer = [0] * 5
        self.sliding_window.data = self.data_buffer[-self.window_size:]

    def _initiate_simulation(self, pbar):
        self.sliding_window.data = self.data_buffer[-self.window_size:]
        get_logger().info(f'Data buffer build of size {self.buffer_size} seconds.')
        get_logger().info(f'--- Starting Simulation ---')
        self.data_buffer_flag = False
        self.data_buffer = self.data_buffer.iloc[self.step_size:]
        self.sliding_window.data = self.sliding_window.data.iloc[self.step_size:]
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
