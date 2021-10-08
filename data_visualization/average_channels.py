from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from classes import Frame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from definitions import OUTPUT_PATH
from utility.logger import get_logger


# Finds the emg_peaks within its corresponding tp interval
def find_usable_emg(tp_table: pd.DataFrame) -> [int]:
    emg = []
    for i in range(0, len(tp_table)):
        if tp_table['tp_start'][i] < tp_table['emg_peak'][i] < tp_table['tp_end'][i]:
            emg.append(i)
        else:
            get_logger().debug('EMG detected outside tp_start-tp_end', tp_table.iloc[i])

    return emg


# takes in list of all frames with MRCP and returns a frame containing the average of them.
def average_channel(frames: [Frame], emg_detections: [int]) -> [Frame]:
    # test = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 19, 20, 24, 28]
    columns = frames[0].filtered_data.columns
    avg_channel = []
    for col in columns:
        df_temp = pd.DataFrame()
        # for i in range(0, len(frames)):
        #     df_temp[i] = frames[i].filtered_data[col]
        for i in emg_detections:
            df_temp[i] = frames[i].filtered_data[col]

        frame = Frame.Frame()
        frame.data = df_temp.mean(axis=1)
        avg_channel.append(frame)

    return avg_channel


def plot_average_channels(avg_channels: [Frame], freq: int = 1200, save_fig: bool = False):
    # linspace: Returns evenly spaced numbers over a specified interval.

    for channel in range(0, len(avg_channels)):
        x = []
        y = []
        center = (len(avg_channels[channel].data) / 2) / freq
        for i, row in avg_channels[channel].data.items():
            x.append(i / freq - center)
            y.append(row)

        y = np.array(y)

        size = len(x)
        x_new = np.linspace(0, size, size)
        y_hat = savgol_filter(y, int(size / 10) + 1, 1)

        plt.plot(x, y)
        plt.axvline(x=0, color='black', ls='--')
        plt.plot(x, y_hat)
        plt.title(f'Channel: {channel + 1}')

        if save_fig:
            plt.savefig(f'{OUTPUT_PATH}/highpass/frames/{channel + 1}.png')

        plt.show()


