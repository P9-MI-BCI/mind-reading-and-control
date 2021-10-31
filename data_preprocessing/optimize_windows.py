# performs a grid search to cost minimize array.
import json
import pandas as pd
import numpy as np

from data_preprocessing.mrcp_detection import mrcp_detection
from data_preprocessing.eog_detection import blink_detection
from data_visualization.average_channels import windows_based_on_heuristic, average_channel
from utility.logger import get_logger
from classes import Window, Dataset
from tqdm import tqdm

'''
find_best_config_params is a deprecated method. It was used for grid search of filter parameters.
'''

# todo change sample indexes to be ids of the windows instead
# test distance from minimum to middle
# what sample has the biggest negative influence on the average

def find_best_config_params(data, trigger_table, config):
    emg_order_range = [4, 5, 6]
    emg_cutoff_range = list(range(75, 110))
    eeg_cutoff_range_min = [0.04, 0.05, 0.06, 0.07]
    eeg_cutoff_range_max = [4, 5, 6, 7, 8, 9, 10]
    eeg_order = [2, 3]

    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    minimize_cost = 99999999
    minimized_config = {}

    for order in tqdm(emg_order_range):
        for cutoff in tqdm(emg_cutoff_range):
            for eeg_min in eeg_cutoff_range_min:
                for eeg_max in eeg_cutoff_range_max:
                    for eeg_ord in eeg_order:
                        config['emg_order'] = order
                        config['emg_cutoff'] = cutoff
                        config['eeg_cutoff'] = [eeg_min, eeg_max]
                        config['eeg_order'] = eeg_ord

                        try:
                            emg_windows, trigger_table = mrcp_detection(data=data, tp_table=trigger_table,
                                                                        config=config)

                            # Find valid emgs based on heuristic and calculate averages
                            valid_emg = [0, 3, 7, 8, 10, 12, 13, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]

                            windows = average_channel(emg_windows, valid_emg)

                            minimize_array = []
                            for i in range(0, len(windows)):
                                if i in channels:
                                    minimize_array.append(abs(windows[i].data.idxmin() - int(len(windows[i].data) / 2)))

                            if sum(minimize_array) < minimize_cost:
                                minimized_config = config
                                minimize_cost = sum(minimize_array)
                                get_logger().debug(f'New shortest distance/cost {minimize_cost}')
                                get_logger().debug(f'config: {config}')
                        except:
                            get_logger().debug(f'During param search - Config : {config} did not work.')
        # for some reason always prints the last params ..
    print(minimized_config)


# Defines a method to score each window depending on their distance to the min y val in the function.
# Higher score is worse.
def optimize_average_minimum(valid_emg: [int], emg_windows: [Window], channels: [int] = None, weights: [int] = None,
                             remove: int = 10) -> [int]:
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    base_windows = average_channel(emg_windows, valid_emg)  # the windows to calc avg of.

    base_score = []
    for i in range(0, len(base_windows)):
        if i in channels:
            base_score.append(abs(base_windows[i].data.idxmin() - int(len(base_windows[i].data) / 2)) * weights[
                i])  # calc the distance from the min y to where the window center is.

    minimize_cost = sum(base_score)

    get_logger().debug(f'Base score is {minimize_cost}')
    # Removes the amount (remove param) of windows with the highest score
    try:
        for rem in range(0, remove):
            worst_sample = None
            get_logger().debug(f'Current Valid EMGs {valid_emg}')

            for sample in range(0, len(valid_emg)):
                minimize_array = []

                # iterate over all combinations of array combination
                b = [x for i, x in enumerate(valid_emg) if i != sample]
                windows = average_channel(emg_windows, b)

                for i in range(0, len(windows)):
                    if i in channels:
                        minimize_array.append(
                            abs(windows[i].data.idxmin() - int(len(windows[i].data) / 2)) * weights[i])

                if sum(minimize_array) <= minimize_cost:
                    worst_sample = sample
                    minimize_cost = sum(minimize_array)
                    get_logger().debug(f'New shortest distance/cost {minimize_cost}')
                    get_logger().debug(f'Attained by removing sample with index {worst_sample}')

            try:
                del valid_emg[worst_sample]
            except TypeError:
                get_logger().exception(f'It was not possible to create a better subset of values. {rem} '
                                       f'values were removed from the array')
                return valid_emg
    except ValueError:
        get_logger().error(f'You are trying to remove more samples than there is valid emgs detected. There are '
                           f'{len(valid_emg)} valid emgs you are trying to remove {rem} more.')

    return valid_emg


def remove_worst_windows(emg_windows: [Window],
                         channels: [int] = None,
                         weights: [int] = None,
                         remove: int = 10) -> [int]:
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    remove_windows = []
    for rem in range(0, remove):

        get_logger().debug(f'Iteration {rem} - Amount of valid windows {len(emg_windows)}')
        idx_ws = 0
        worst_sample = 0
        for sample in range(0, len(emg_windows)):
            minimize_array = []
            for chan in range(0, len(channels)):
                center = int(len(emg_windows[sample].filtered_data[chan]) / 2)
                minimize_array.append(
                    abs(emg_windows[sample].filtered_data[chan].idxmin() - center) * weights[chan])

            if sum(minimize_array) > worst_sample:
                worst_sample = sum(minimize_array)
                idx_ws = sample
                get_logger().debug(
                    f'Worst sample: {[idx_ws]} - with score: {worst_sample} - based on channels {channels}')

        try:
            remove_windows.append(emg_windows[idx_ws])
            del emg_windows[idx_ws]
        except:
            get_logger().exception(f'It was not possible to create a better subset of values. {rem} '
                                   f'values were removed from the array')
            return remove_windows

    # get_logger().debug(f'Resulting array: {valid_emg}')
    return remove_windows


def prune_poor_quality_samples(windows: [Window], trigger_table: pd.DataFrame, config,
                               remove: int = 8, method=None, heuristic: bool = False):
    # Find valid emgs based on heuristic and calculate averages
    mrcp_windows = []
    for window in windows:
        if window.label == 1 and not window.is_sub_window:
            mrcp_windows.append(window)

    if heuristic:
        valid_emg = windows_based_on_heuristic(trigger_table, config)  # Our simple time heuristic

    # remove = remove - (mrcp_windows - len(valid_emg))
    if remove < 0:
        get_logger().error('You are trying to remove more windows than exists')
        return

    if method:
        # method here can be either optimize_average_minimum or remove_worst_windows
        remove_mrcp = method(mrcp_windows, remove=remove)

    # deletes windows from last index first, in order to avoid index collision
    del_list = []

    for remove_w in remove_mrcp:
        for iw, w in enumerate(windows):
            if remove_w.num_id == w.num_id:
                del_list.append(iw)
                # delete sub windows then delete rest
                for sub_w in remove_w.sub_windows:
                    for i, sw in enumerate(windows):
                        if sub_w == sw.num_id:
                            del_list.append(i)
                            break
                rest_id = w.num_id.replace('mrcp', 'rest')
                for ir, wr in enumerate(windows):
                    if wr.num_id == rest_id:
                        del_list.append(ir)
                        for sub_w in wr.sub_windows:
                            for i, sw in enumerate(windows):
                                if sub_w == sw.num_id:
                                    del_list.append(i)
                                    break

    del_list.sort(reverse=True)
    windows.reverse()
    for i in range(len(windows)-1, -1, -1):
        if i in del_list:
            del windows[i]


def remove_windows_with_blink(data: pd.DataFrame, windows: [Window], sample_rate: int = 1200):
    blinks = blink_detection(data=data, sample_rate=sample_rate)

    # Perform blink removal. Checks if any blink frequencies are within any of the windows. Remove window if so.
    window_ids = []
    for index, window in enumerate(windows):
        temp_freq_range = list(range(window.frequency_range[0], window.frequency_range[1]))
        for blink in blinks:
            if blink in temp_freq_range:
                window_ids.append(window.num_id)
                get_logger().debug(f'Blink detected in {window.num_id}, window has label {window.label}')
    for id in window_ids:
        for index, window in enumerate(windows):
            if window.num_id == id:
                del windows[index]
