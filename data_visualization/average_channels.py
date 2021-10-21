import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import OUTPUT_PATH
from utility.logger import get_logger
from utility.file_util import create_dir, file_exist
from utility.save_figure import save_figure
from classes import Window


# Finds the emg_peaks within its corresponding tp interval
def find_usable_emg(tp_table: pd.DataFrame, config) -> [int]:
    emg = []
    for i in range(0, len(tp_table)):
        if tp_table['tp_start'][i] < tp_table[config['aggregate_strategy']][i] < tp_table['tp_end'][i]:
            emg.append(i)
        else:
            get_logger().debug('EMG detected outside tp_start-tp_end')

    return emg


# takes in list of all windows with MRCP and returns a window containing the average of them.
def average_channel(windows: [Window]) -> [Window]:
    mrcp_windows = 0
    for window in windows:
        if window.label == 1:
            mrcp_windows += 1

    EEG_CHANNELS = list(range(0, 10))
    avg_channel = []
    for col in EEG_CHANNELS:
        df_temp = pd.DataFrame()
        for i in range(0, mrcp_windows):
            df_temp[i] = windows[i].filtered_data[col]

        window = Window.Window()
        window.data = df_temp.mean(axis=1)
        avg_channel.append(window)

    return avg_channel


def plot_average_channels(avg_channels: [Window], freq: int = 1200, save_fig: bool = False, overwrite: bool = False):
    for channel in range(0, len(avg_channels)):
        x = []
        y = []
        center = (len(avg_channels[channel].data) / 2) / freq
        for i, row in avg_channels[channel].data.items():
            x.append(i / freq - center)
            y.append(row)

        y = np.array(y)

        fig = plt.figure()
        plt.plot(x, y)
        plt.axvline(x=0, color='black', ls='--')
        plt.title(f'Channel: {channel + 1}')

        if save_fig:
            path = f'{OUTPUT_PATH}/plots/average_emg_start/{channel + 1}.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=overwrite)
            except FileExistsError:
                get_logger().exception(f'Found file already exists: {file} you can '
                                       f'overwrite the file by setting overwrite=True')

        plt.show()
