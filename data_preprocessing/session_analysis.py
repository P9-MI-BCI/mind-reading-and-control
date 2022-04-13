import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import OrderedDict
from classes import Dataset
from definitions import OUTPUT_PATH
from utility.save_figure import save_figure
from utility.logger import get_logger


def session_analysis_hub(training_data, config, subject_id):
    mean, var, rsd = session_analysis(training_data)  # Mean and var over all sessions.
    # std_mean, median_mean = calc_std_median(mean)  # std and median of each channel over all sessions
    # std_var, median_var = calc_std_median(var)  # --"--

    boxplot(rsd, config, subject_id, 'rsd', show_points=False, save_fig=True)
    barchart(rsd, config, subject_id, 'rsd', save_fig=True)


"""
    Calc the mean and var values for each channel in each session.
        session_mean = [mean_ch0, mean_ch1, ..., mean_ch8]
        session_var = [var_ch0, var_ch1, ..., var_ch8] 
    Returns:
        subject_mean = [session_mean1, ..., session_mean20]
        subject_var = [session_var1, ..., session_var20]
"""


def session_analysis(training_data: [Dataset]):
    subject_mean = []
    subject_var = []
    subject_rsd = []
    # Goes through the 20 sessions of current subject
    for session_data in training_data:
        session_data.data.drop(columns=['EMG'], inplace=True)  # Removes EMG channel
        session_mean = []
        session_var = []
        session_rsd = []
        # Goes through each channel in each session
        for channel in session_data.data.columns:
            session_mean.append(session_data.data[channel].mean())
            session_var.append(session_data.data[channel].var())

            rsd = relative_standard_deviation(session_data.data[channel].to_list())
            session_rsd.append(rsd)

        subject_mean.append(session_mean)
        subject_var.append(session_var)
        subject_rsd.append(session_rsd)

    return subject_mean, subject_var, subject_rsd


#  Calculates the std and median of each channel over ALL sessions.
def calc_std_median(session_res):
    channelwise_std = []
    channelwise_median = []
    for i in range(len(session_res[0])):
        channel_i = [channel_res[i] for channel_res in session_res]
        channelwise_std.append(np.std(channel_i))
        channelwise_median.append(np.median(channel_i))

    return channelwise_std, channelwise_median


def relative_standard_deviation(single_channel_data):
    std = np.std(single_channel_data)
    mean = np.mean(single_channel_data)
    rsd = std / abs(mean)

    return rsd


def convert_to_percentage(channelwise_median):
    median = np.median(channelwise_median)

    percent_values = list(map(lambda val: (val / median) * 100, channelwise_median))

    return percent_values


def boxplot(session_results, config, subject_id: int, measure: str, show_points: bool = True, save_fig: bool = False):
    fig, ax = plt.subplots()

    # Boxplot
    for i in range(len(session_results[0])):  # len 9
        channel_i = [channel_res[i] for channel_res in session_results]
        ax.boxplot(channel_i, positions=[i], showfliers=False)
        # Data points plot
        if show_points:
            for idx, val in enumerate(channel_i):
                ax.plot(i, val, marker='o', label=f'Session {idx}')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 8})

    plt.xticks(range(len(config.EEG_CHANNELS)), config.EEG_CHANNELS)
    plt.xlabel('Channels')
    plt.ylabel(measure.upper())
    plt.title(f'Subject {subject_id}')

    if save_fig:
        path = os.path.join(OUTPUT_PATH, 'session_analysis', f'subject_{subject_id}', f'boxplot_{measure}.png')
        file = os.path.split(path)[1]
        try:
            save_figure(path, fig, overwrite=True)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')
    plt.show()


def barchart(session_results, config, subject_id: int, measure: str, save_fig: bool = False):
    df = pd.DataFrame(session_results, columns=config.EEG_CHANNELS)
    barchart = df.plot(y=config.EEG_CHANNELS, kind="bar", rot=0)
    plt.title(f'Subject {subject_id}')
    plt.xlabel('Session')
    plt.ylabel('RSD')

    if save_fig:
        path = os.path.join(OUTPUT_PATH, 'session_analysis', f'subject_{subject_id}', f'barchart_{measure}.png')
        file = os.path.split(path)[1]
        try:
            fig = barchart.get_figure()
            save_figure(path, fig, overwrite=True)
        except FileExistsError:
            get_logger().exception(f'Found file already exists: {file} you can '
                                   f'overwrite the file by setting overwrite=True')
    plt.show()
