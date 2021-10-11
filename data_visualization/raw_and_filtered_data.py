import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import chain
from utility.save_figure import save_figure
from utility.logger import get_logger
from definitions import OUTPUT_PATH


def plot_raw_filtered_data(data: pd.DataFrame, save_fig: bool = False):

    # Selecting eeg channel 0-8 and emg channel 13
    channels = list(range(0, 13))
    channels[9:12] = []

    for channel in channels:
        fig = plt.figure()
        plt.plot(data.data_device1[channel], label='Raw data')
        plt.plot(data.filtered_data[channel], label='Filtered data')
        if channel < 9:
            plt.title(f'EEG Channel {channel + 1} - bandpass')
        elif channel == 12:
            plt.title(f'EMG Channel {channel + 1} - highpass')
        plt.legend()

        if save_fig:
            path = f'{OUTPUT_PATH}/plots/raw_filtered_data/channel{channel + 1}.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=False)
            except FileExistsError:
                get_logger().debug(get_logger().error(f'Found file already exists: {file} you can '
                                                      f'overwrite the file by setting overwrite=True'))

    plt.show()
