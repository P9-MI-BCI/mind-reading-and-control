#from utility.save_figure import save_figure
from utility.logger import get_logger
from definitions import OUTPUT_PATH
import matplotlib.pyplot as plt
import os
import pandas as pd


class Dataset:

    def __init__(self, handle_arrow_rand=0, no_movements=0, time_cue_on=0, time_cue_off=0, TriggerPoint=0, delay_T1=0,
                 delay_random_T1=0, delay_T2=0, sample_rate=0, time_window=0, no_time_windows=0, filter_code_eeg=0,
                 time_start_device1=0, time_after_first_window=0, time_after_last_window=0, time_stop_device1=0,
                 data_device1=pd.DataFrame, time_axis_all_device1=0, filtered_data=pd.DataFrame):
        self.handle_arrow_rand = handle_arrow_rand
        self.no_movements = no_movements
        self.time_cue_on = time_cue_on
        self.time_cue_off = time_cue_off
        self.TriggerPoint = TriggerPoint
        self.delay_T1 = delay_T1
        self.delay_random_T1 = delay_random_T1
        self.delay_T2 = delay_T2
        self.sample_rate = sample_rate
        self.time_window = time_window
        self.no_time_windows = no_time_windows
        self.filter_code_eeg = filter_code_eeg
        self.time_start_device1 = time_start_device1
        self.time_after_first_window = time_after_first_window
        self.time_after_last_window = time_after_last_window
        self.time_stop_device1 = time_stop_device1
        self.data_device1 = data_device1
        self.time_axis_all_device1 = time_axis_all_device1
        self.filtered_data = filtered_data  # experimental be careful of use

    def plot(self, save_fig=False, overwrite=False):
        # Selecting eeg channel 0-8 and emg channel 12
        channels = self.filtered_data.columns

        for channel in channels:
            fig = plt.figure()
            plt.plot(self.data_device1[channel], label='Raw data')
            plt.plot(self.filtered_data[channel], label='Filtered data')
            if channel < 9:
                plt.title(f'EEG Channel {channel + 1} - bandpass')
            elif channel == 12:
                plt.title(f'EMG Channel {channel + 1} - highpass')
            plt.legend()

            if save_fig:
                path = f'{OUTPUT_PATH}/plots/raw_filtered_data/channel{channel + 1}.png'
                file = os.path.split(path)[1]
                try:
                    pass
                    # save_figure(path, fig, overwrite=overwrite)
                except FileExistsError:
                    get_logger().exception(f'Found file already exists: {file} you can '
                                           f'overwrite the file by setting overwrite=True')

        plt.show()
