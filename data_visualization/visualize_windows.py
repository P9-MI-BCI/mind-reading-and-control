import os
import numpy as np
import matplotlib.pyplot as plt
from classes import Dataset, Window
from definitions import OUTPUT_PATH
from utility.save_figure import save_figure
from utility.logger import get_logger


# Visualizes the windows that are cut
def visualize_windows(data: Dataset, windows: [Window], channel: int = 4, xlim: int = 400000, savefig: bool = False,
                      overwrite: bool = False):
    fig = plt.figure()

    for window in windows:
        start = window.frequency_range[0]
        end = window.frequency_range[-1]
        window_span = plt.axvspan(start, end, color='grey', alpha=0.5, label='Windows')

    plt.xlabel('Frequency')
    filtered_data, = plt.plot(data.filtered_data[channel], label='Filtered data')
    plt.xlim(0, xlim)
    plt.legend(handles=[filtered_data, window_span])
    plt.title(f'Windows of Channel: {channel + 1}')

    if savefig:
        path = f'{OUTPUT_PATH}/plots/window_overview/{channel + 1}.png'
        file = os.path.split(path)[1]
        try:
            save_figure(path, fig, overwrite=overwrite)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')
    plt.show()


# Visualizes the windows that are cut with their associated labels
def visualize_labeled_windows(data: Dataset, windows: [Window], channel: int = 4, xlim: int = 400000,
                              savefig: bool = False, overwrite: bool = False):
    fig, ax = plt.subplots(figsize=(10,6))

    for window in windows:
        start = window.frequency_range[0]
        end = window.frequency_range[-1]

        if window.label == 1 and window.blink == 0:
            mrcp_win = plt.axvspan(start, end, color='green', alpha=0.5, label='MRCP')
        elif window.label == 1 and window.blink == 1:
            mrcp_blink_win = plt.axvspan(start, end, color='red', alpha=0.5, label='MRCP w. blink')
        elif window.label == 0 and window.blink == 0:
            idle_win = plt.axvspan(start, end, color='grey', alpha=0.5, label='Rest')
        elif window.label == 0 and window.blink == 1:
            idle_blink_win = plt.axvspan(start, end, color='orange', alpha=0.5, label='Rest w. blink')

    plt.xlabel('Frequency')
    filtered_data, = plt.plot(data.filtered_data[channel], label='Filtered data')
    plt.xlim(0, xlim)
    plt.legend(handles=[filtered_data, mrcp_win, mrcp_blink_win, idle_win, idle_blink_win], loc='center',
               bbox_to_anchor=(1.15, 0.5))
    plt.title(f'Labeled Windows of Channel: {channel + 1}')
    fig.tight_layout()

    if savefig:
        path = f'{OUTPUT_PATH}/plots/window_overview/{channel + 1}_labeled.png'
        file = os.path.split(path)[1]
        try:
            save_figure(path, fig, overwrite=overwrite)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')
    plt.show()


# Visualizes a single window for all channels
def visualize_window_all_channels(filtered_data, window: Window, window_id: int, savefig: bool = False,
                                  overwrite: bool = False):
    fig = plt.figure(figsize=(12, 10))

    start = 0
    end = 2400

    for channel in range(9):
        ax = fig.add_subplot(3, 3, channel + 1)
        ax.axvspan(start, end, color='grey', alpha=0.5)
        ax.set_title(f'Channel {channel + 1}')
        ax.plot(window.filtered_data[channel])
        if channel == 3:
            ax.set_ylabel('μV (Filtered)')
        if channel == 7:
            ax.set_xlabel('time (s)')

    plt.tight_layout()

    if savefig:
        path = f'{OUTPUT_PATH}/plots/window_overview/all_channels_single_window.png'
        file = os.path.split(path)[1]
        try:
            save_figure(path, fig, overwrite=overwrite)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')

    plt.show()
