from tqdm import tqdm

from classes.Dataset import Dataset
from classes.Window import Window
import time
import pandas as pd
from utility.logger import get_logger
from data_preprocessing.filters import butter_filter
import datetime


def _dataset_information(dataset: Dataset):
    get_logger().info(
        f'Dataset takes estimated: '
        f'{datetime.timedelta(seconds=round(len(dataset.data_device1) / dataset.sample_rate))} to simulate.')
    get_logger().info(f'Dataset is sampled at: {dataset.sample_rate} frequency.')
    get_logger().info(f'Dataset contains: {(len(dataset.TriggerPoint)) // 2} cue onsets.')


class Simulation:

    def __init__(self, config, dataset: Dataset = 0):
        self.time = time.time()

        if dataset:
            self.dataset = dataset

        if config:
            self.window_size = config.window_size
            self.buffer_size = self.window_size * config.buffer_size
            self.step_size = config.step_size
            self.EEG_channels = config.EEG_Channels
            self.data_buffer = pd.DataFrame(columns=self.EEG_channels)

            self.filtering = config.filtering
            self.EEG_order = config.eeg_order
            self.EEG_cutoff = config.eeg_cutoff
            self.EEG_btype = config.eeg_btype

            self.cue_set = config.id

    def mount_dataset(self, dataset: Dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.window_size = self.window_size * dataset.sample_rate
        self.step_size = int(self.step_size * dataset.sample_rate)

    def simulate(self, real_time: bool, description: bool = True):
        assert bool(self.dataset)

        if description:
            self._simulation_information()

        i = 0
        if not real_time:
            get_logger().info('---------- ########## ----------')
            get_logger().info('Real time simulation disabled.')
        else:
            get_logger().info('---------- ########## ----------')
            get_logger().info('Real time simulation enabled.')
            time_pass = self.step_size / self.dataset.sample_rate
        data_buff_checkmark = True
        get_logger().info('Building Data Buffer.')
        with tqdm(total=len(self.dataset.data_device1)) as pbar:
            with tqdm(total=self.buffer_size * self.window_size, position=0) as bpbar:
                sliding_window = Window()
                while i < len(self.dataset.data_device1):
                    self.time = time.time()

                    try:
                        sliding_window.data = pd.concat(
                            [sliding_window.data.iloc[i:], self.dataset.data_device1.iloc[i: i + self.step_size]])
                    except AttributeError:
                        pass
                    if len(self.data_buffer) < self.buffer_size * self.window_size and data_buff_checkmark:
                        self.data_buffer = pd.concat([self.data_buffer, self.dataset.data_device1.iloc[i:i + self.step_size]],
                                                     ignore_index=True)
                        bpbar.update(self.step_size)
                        if real_time:
                            time_after = time.time()
                            elapsed_time = time_after - self.time
                            if time_pass - elapsed_time > 0:
                                time.sleep(time_pass - elapsed_time)
                            else:
                                get_logger().warning(
                                    f'The previous iteration too more than {time_pass} '
                                    f'to process - increasing sliding window distance.')
                                self.step_size += 60
                            i += self.step_size
                        else:
                            i += self.step_size
                        continue

                    if data_buff_checkmark:
                        sliding_window.data = self.data_buffer[-self.window_size:]
                        get_logger().info(f'Data buffer build of size {self.buffer_size} windows.')
                        get_logger().info(f'--- Starting Simulation ---')
                        data_buff_checkmark = False
                        pbar.update(i)
                    else:
                        self.data_buffer = pd.concat([self.data_buffer, sliding_window.data[-self.step_size:]],
                                                     ignore_index=True)

                    if self.filtering:
                        filtered_data = pd.DataFrame(columns=self.EEG_channels)

                        for channel in self.EEG_channels:
                            filtered_data[channel] = butter_filter(data=self.data_buffer[channel],
                                                                   order=self.EEG_order,
                                                                   cutoff=self.EEG_cutoff,
                                                                   btype=self.EEG_btype
                                                                   )

                    # deallocate data
                    self.data_buffer = self.data_buffer.iloc[self.step_size:]
                    sliding_window.data = sliding_window.data.iloc[self.step_size:]
                    if real_time:
                        time_after = time.time()
                        elapsed_time = time_after - self.time
                        if time_pass - elapsed_time > 0:
                            time.sleep(time_pass - elapsed_time)
                        else:
                            get_logger().warning(
                                f'The previous iteration too more than {time_pass} '
                                f'to process - increasing sliding window distance.')
                            self.step_size += 60
                        i += self.step_size
                    else:
                        i += self.step_size
                    pbar.update(self.step_size)

    def feature_extraction(self, method):
        # provide method or use existing
        pass

    def metrics(self, metrics, concurrently: bool = False):
        for metric in metrics:
            pass  # implement each metric given in arr
        pass

    def _simulation_information(self):
        get_logger().info('-- # Simulation Description # --')
        get_logger().info(f'Window size: {self.window_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'Initial step size : {self.step_size / self.dataset.sample_rate} seconds')
        get_logger().info(f'Buffer size: {self.buffer_size} seconds')
        get_logger().info(f'EEG Channels: {self.EEG_channels}')
        get_logger().info(f'Filtering: {bool(self.filtering)}')
        get_logger().info(f'Dataset ID: {self.cue_set}')
        _dataset_information(self.dataset)
