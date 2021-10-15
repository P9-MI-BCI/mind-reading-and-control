import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from classes import Window
from definitions import OUTPUT_PATH
from utility.save_figure import save_figure
from utility.logger import get_logger


def visualize_window(window: Window, config, freq: int, channel: int, num: int, save_fig: bool = False,
                    overwrite: bool = False):
    x_seconds = []
    fig = plt.figure(figsize=(5, 7))
    center = (len(window.data) / 2) / freq
    for i, row in window.filtered_data[channel].items():  # converts the window.data freqs to seconds
        x_seconds.append(i / freq - center)

    agg_strat = config['aggregate_strategy']

    if window.label == 1:
        if agg_strat == 'emg_start':
            emg_timestamp = [0, (
                    window.timestamp['emg_peak'] - window.timestamp[agg_strat]).total_seconds(),
                             (window.timestamp['emg_end'] - window.timestamp[
                                 agg_strat]).total_seconds()]
            tp_timestamp = [
                (window.timestamp['tp_start'] - window.timestamp[agg_strat]).total_seconds(),
                (window.timestamp['tp_end'] - window.timestamp[agg_strat]).total_seconds()]

        elif agg_strat == 'emg_peak':
            emg_timestamp = [
                (window.timestamp['emg_start'] - window.timestamp[agg_strat]).total_seconds(), 0,
                (window.timestamp['emg_end'] - window.timestamp[agg_strat]).total_seconds()]
            tp_timestamp = [
                (window.timestamp['tp_start'] - window.timestamp[agg_strat]).total_seconds(),
                (window.timestamp['tp_end'] - window.timestamp[agg_strat]).total_seconds()]
        elif agg_strat == 'emg_end':
            emg_timestamp = [
                (window.timestamp['emg_start'] - window.timestamp[agg_strat]).total_seconds(),
                (window.timestamp['emg_peak'] - window.timestamp[agg_strat]).total_seconds(), 0]
            tp_timestamp = [
                (window.timestamp['tp_start'] - window.timestamp[agg_strat]).total_seconds(),
                (window.timestamp['tp_end'] - window.timestamp[agg_strat]).total_seconds()]
        y_t = ['TP'] * len(tp_timestamp)
        y_t2 = ['EMG'] * len(emg_timestamp)

        gs = gridspec.GridSpec(ncols=1, nrows=6, figure=fig)
        ax1 = fig.add_subplot(gs[:2, 0])
        ax1.set_title(f' Channel: {channel + 1} - EEG {num + 1} - Filter: No Filter')
        ax1.plot(x_seconds, window.data[channel], color='tomato')
        ax1.axvline(x=0, color='black', ls='--')

        ax4 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
        ax4.set_title(f'Filter: {window.filter_type[channel].iloc[0]}')
        ax4.plot(x_seconds, window.filtered_data[channel], color='tomato')
        ax4.axvline(x=0, color='black', ls='--')

        ax2 = fig.add_subplot(gs[4, 0], sharex=ax1)
        ax2.set_title('EMG Detection')
        ax2.plot(emg_timestamp, y_t2, marker='^', color='limegreen')
        ax2.annotate('Peak', xy=[emg_timestamp[1], y_t2[1]])

        ax3 = fig.add_subplot(gs[5, 0], sharex=ax1)
        ax3.set_title('Trigger Point Duration')
        ax3.plot(tp_timestamp, y_t, marker='o', color='royalblue')
        ax3.annotate('Trigger Point', xy=[tp_timestamp[0], y_t[0]])


    else:
        plt.title(f' Channel: {channel + 1} - EEG Window: {num + 1} - Filter: {window.filter_type[channel].iloc[0]}')
        plt.plot(x_seconds, window.filtered_data[channel], color='tomato')
        plt.axvline(x=0, color='black', ls='--')

    plt.tight_layout()

    if save_fig:
        path = f'{OUTPUT_PATH}/plots/window_plots/channel{channel}_{num + 1}.png'
        file = os.path.split(path)[1]
        try:
            save_figure(path, fig, overwrite=overwrite)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')

    plt.show()
