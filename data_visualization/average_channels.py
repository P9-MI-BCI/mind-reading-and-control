import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import OUTPUT_PATH
from utility.logger import get_logger
from utility.file_util import create_dir, file_exist
from utility.save_figure import save_figure
from classes import Frame


# Finds the emg_peaks within its corresponding tp interval
def find_usable_emg(tp_table: pd.DataFrame, config) -> [int]:
    emg = []
    for i in range(0, len(tp_table)):
        if tp_table['tp_start'][i] < tp_table[config['aggregate_strategy']][i] < tp_table['tp_end'][i]:
            emg.append(i)
        else:
            get_logger().debug('EMG detected outside tp_start-tp_end', tp_table.iloc[i])

    return emg


# takes in list of all frames with MRCP and returns a frame containing the average of them.
def average_channel(frames: [Frame], emg_detections: [int]) -> [Frame]:
    EEG_CHANNELS = list(range(0, 9))
    avg_channel = []
    for col in EEG_CHANNELS:
        df_temp = pd.DataFrame()
        for i in emg_detections:
            df_temp[i] = frames[i].filtered_data[col]

        frame = Frame.Frame()
        frame.data = df_temp.mean(axis=1)
        avg_channel.append(frame)

    return avg_channel


def plot_average_channels(avg_channels: [Frame], freq: int = 1200, save_fig: bool = False, overwrite: bool = False):
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
                get_logger().debug(get_logger().error(f'Found file already exists: {file} you can '
                                                      f'overwrite the file by setting overwrite=True'))

        plt.show()
