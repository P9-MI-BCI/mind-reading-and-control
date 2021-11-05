from tqdm import tqdm

from classes.Dataset import Dataset
from classes.Window import Window
import time
import pandas as pd
from utility.logger import get_logger
from data_preprocessing.filters import butter_filter
import datetime


def _dataset_information(dataset: Dataset):
    get_logger().info(f'Dataset takes {datetime.timedelta(seconds=len(dataset.data_device1)/dataset.sample_rate)} to consume.')
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

    def mount_dataset(self, dataset: Dataset, verbose: int = 1):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.window_size = self.window_size * dataset.sample_rate

        if verbose == 1:
            _dataset_information(dataset)
        # log information regarding dataset

    def simulate(self, real_time: bool):
        assert bool(self.dataset)

        i = 0
        self.time = time.time()
        if not real_time:
            get_logger().info('Real time simulation disabled.')
        with tqdm(total=len(self.dataset.data_device1)) as pbar:
            while i < len(self.dataset.data_device1):
                sliding_window = Window()
                sliding_window.data = self.dataset.data_device1.iloc[i: i + self.step_size]

                if len(self.data_buffer) < self.buffer_size:
                    self.data_buffer = pd.concat([self.data_buffer, sliding_window.data], ignore_index=True)
                    continue

                self.data_buffer = pd.concat([self.data_buffer, sliding_window.data], ignore_index=True)

                if self.filtering:
                    filtered_data = pd.DataFrame(columns=self.EEG_channels)

                    for channel in self.EEG_channels:
                        filtered_data[channel] = butter_filter(data=self.data_buffer[channel],
                                                               order=self.EEG_order,
                                                               cutoff=self.EEG_cutoff,
                                                               btype=self.EEG_btype
                                                               )

                self.data_buffer = self.data_buffer.iloc[self.step_size:]

                if real_time:
                    # calculate time after passover and sleep(?)
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
