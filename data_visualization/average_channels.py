import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import OUTPUT_PATH
from utility.logger import get_logger
from utility.file_util import create_dir, file_exist
from utility.save_figure import save_figure
from classes import Window


# Finds the emg_peaks within its corresponding tp interval (our simple heuristic)
def windows_based_on_heuristic(tp_table: pd.DataFrame, config) -> [int]:
    windows = []
    agg_strat = config['aggregate_strategy']
    for i in range(0, len(tp_table)):
        if tp_table['tp_start'][i] < tp_table[agg_strat][i] < tp_table['tp_end'][i]:
            windows.append(i)
        else:
            get_logger().debug(f'{agg_strat} detected outside tp_start-tp_end')

    return windows


# Takes all windows with MRCP within each EEG channel and calculates the mean of them.
def average_channel(windows: [Window], mrcp_windows: [int] = None) -> [Window]:
    if mrcp_windows is None:
        i = 0
        mrcp_windows = []
        for window in windows:
            if window.label == 1: # and not window.is_sub_window:
                mrcp_windows.append(i)
            i += 1

    EEG_CHANNELS = list(range(0, 9))
    avg_channel = []
    for col in EEG_CHANNELS:
        df_temp = pd.DataFrame()
        for i in mrcp_windows:
            df_temp[i] = windows[i].filtered_data[col]

        window = Window.Window()
        window.data = df_temp.mean(axis=1)
        avg_channel.append(window)

    return avg_channel


def plot_average_channels(avg_channels: [Window], config, freq: int = 1200, layout: str = 'separate',
                          save_fig: bool = False, overwrite: bool = False, weights = None):

    agg_strat = config['aggregate_strategy']
    fig = plt.figure(figsize=(10,10))
    for channel in range(0, len(avg_channels)):
        x = []
        y = []
        center = (len(avg_channels[channel].data) / 2) / freq
        for i, row in avg_channels[channel].data.items():
            x.append(i / freq - center)
            y.append(row)

        y = np.array(y)

        # Makes individual plots of each eeg channel
        if layout == 'separate':
            fig = plt.figure()
            plt.axvline(x=0, color='black', ls='--')
            # plt.title(f'Channel: {channel + 1} - Agg. Strat: {agg_strat}')
            plt.ylabel('μV (Filtered)')
            # if channel == 7:
            plt.xlabel('time (s)')
            plt.plot(x, y)

        # Makes a grid plot of all eeg channels
        elif layout == 'grid':
            ax = fig.add_subplot(3, 3, channel + 1)
            plt.axvline(x=0, color='black', ls='--')
            if weights is None:
                ax.set_title(f'Channel {channel + 1}')
            else:
                ax.set_title(f'Channel {channel +1} - Weight {round(weights[channel], 2)}')
            if channel == 3:
                ax.set_ylabel('μV (Filtered)')
            if channel == 7:
                ax.set_xlabel('time (s)')
            ax.plot(x, y)
            plt.tight_layout()

        if save_fig:
            if layout == 'separate':
                path = f'{OUTPUT_PATH}/plots/average_{agg_strat}/{channel + 1}.png'
            elif layout == 'grid':
                path = f'{OUTPUT_PATH}/plots/average_{agg_strat}/all_channels.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=overwrite)
            except FileExistsError:
                get_logger().exception(f'Found file already exists: {file} you can '
                                       f'overwrite the file by setting overwrite=True')

    plt.show()
