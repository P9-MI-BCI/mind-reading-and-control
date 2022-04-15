import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from collections import OrderedDict
from classes import Dataset
from definitions import OUTPUT_PATH
from utility.save_figure import save_figure
from utility.logger import get_logger


def session_analysis_hub(training_data, online_data, dwell_data, config, subject_id, save_res: bool = False):
    extend_data = True

    # Mean, var, and rsd over each channel of all sessions.
    mean, var, rsd = session_analysis(training_data, online_data, dwell_data, extend_data=extend_data)

    # std, rsd, and median (of the mean) of each channel over all sessions
    std_mean, rsd_mean, median_mean = calc_std_rsd_median(mean)

    # std, rsd, and median (of the var) of each channel over all sessions
    std_var, rsd_var, median_var = calc_std_rsd_median(var)

    # Plotting and saving results
    if save_res:
        boxplot(rsd, config, subject_id, 'rsd', show_points=True, extend_data=extend_data, save_fig=save_res)
        barchart(rsd, config, subject_id, 'rsd', save_fig=save_res)
        write_results_to_csv(rsd_mean, subject_id, config)


"""
    Calc the mean, var, and RSD values for each channel in each session.
        session_mean = [mean_ch0, mean_ch1, ..., mean_ch8]
        session_var = [var_ch0, var_ch1, ..., var_ch8] 
        session_RSD = [RSD_ch0, RSD_ch1, ..., RSD_ch8] 
    Returns:
        subject_mean = [session_mean1, ..., session_meanN]
        subject_var = [session_var1, ..., session_varN]
        subject_RSD = [session_RSD1, ..., session_RSDN]
        N is 20 (training sessions) if extend_data=False. 23 if True (online 1 & 2 + dwell added) 
"""


def session_analysis(training_data: [Dataset], online_data: [Dataset], dwell_data: Dataset,
                     extend_data: bool = False):
    if extend_data:
        training_data.extend(online_data)
        training_data.append(dwell_data)

    subject_mean = []
    subject_var = []
    subject_rsd = []
    # Goes through the sessions of current subject (if extend_data = True online and dwell sessions are included)
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


#  Calculates the std, rsd, and median of the mean of each channel over ALL sessions.
def calc_std_rsd_median(session_res):
    channelwise_std = []
    channelwise_rsd = []
    channelwise_median = []
    for i in range(len(session_res[0])):
        channel_i = [channel_res[i] for channel_res in session_res]
        channelwise_std.append(np.std(channel_i))
        channelwise_rsd.append(relative_standard_deviation(channel_i))
        channelwise_median.append(np.median(channel_i))

    return channelwise_std, channelwise_rsd, channelwise_median


def relative_standard_deviation(single_channel_data) -> float:
    std = np.std(single_channel_data)
    mean = np.mean(single_channel_data)
    rsd = std / abs(mean)

    return rsd


def convert_to_percentage(channelwise_median):
    median = np.median(channelwise_median)
    percent_values = list(map(lambda val: (val / median) * 100, channelwise_median))

    return percent_values


# show_points decides whether the individual data points of each boxplot is shown
def boxplot(session_results, config, subject_id: int, measure: str, show_points: bool = False,
            extend_data: bool = False, save_fig: bool = False):
    fig, ax = plt.subplots()

    # Boxplot

    for i in range(len(config.EEG_CHANNELS)):
        channel_i = [channel_res[i] for channel_res in session_results]
        ax.boxplot(channel_i, positions=[i], showfliers=False)
        # Data points plot
        if show_points:
            counter = 1
            for idx, val in enumerate(channel_i):
                if extend_data:
                    if idx in {20, 21, 22}:
                        if counter == 1:
                            ax.plot(i, val, marker='o', ms=4, color='b', label=f'Online Test {counter}')
                        elif counter == 2:
                            ax.plot(i, val, marker='o', ms=4, color='r', label=f'Online Test {counter}')
                        else:
                            ax.plot(i, val, marker='o', ms=4, color='g', label=f'Dwell Data')
                        counter += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 8})

    plt.xticks(range(len(config.EEG_CHANNELS)), config.EEG_CHANNELS)
    plt.xlabel('Channels')
    plt.ylabel(measure)
    plt.title(f'Subject {subject_id}')

    if save_fig:
        if extend_data:
            path = os.path.join(OUTPUT_PATH, 'session_analysis', f'subject_{subject_id}', f'boxplot_{measure}.png')
        else:
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

    labels = [item.get_text() for item in barchart.get_xticklabels()]
    labels[-3:] = ['Test 1', 'Test 2', 'Dwell']  # changes the labels of the last three xticks
    barchart.set_xticklabels(labels)
    for label in barchart.get_xticklabels()[-3:]:
        label.set_rotation(30)

    plt.xlabel('Session', labelpad=-10)
    plt.ylabel('RSD')
    plt.title(f'Subject {subject_id}')
    plt.tight_layout()

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


def write_results_to_csv(result: [float], subject_id: int, config):
    path = os.path.join(OUTPUT_PATH, 'session_analysis', 'analysis_results.csv')
    header = ['subject_id'] + config.EEG_CHANNELS

    try:
        with open(path, 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # if file does not exist
                writer.writerow(header)
            result = [round(val, 3) for val in result]
            result = [subject_id] + result
            writer.writerow(result)
    except EnvironmentError as error:  # parent of IOError, OSError *and* WindowsError where available
        get_logger().exception(error)
