# performs a grid search to cost minimize array.
import json
import pandas as pd
import numpy as np

from data_preprocessing.mrcp_detection import mrcp_detection
from data_preprocessing.eog_detection import blink_detection
from data_visualization.average_channels import windows_based_on_heuristic, average_channel
from utility.logger import get_logger
from classes import Window
from tqdm import tqdm


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


# todo fix
def optimize_average_minimum(valid_emg, emg_windows, channels=None, weights=None, remove: int = 10):
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    base_windows = average_channel(emg_windows, valid_emg)

    base_score = []
    for i in range(0, len(base_windows)):
        if i in channels:
            base_score.append(abs(base_windows[i].data.idxmin() - int(len(base_windows[i].data) / 2)))

    minimize_cost = sum(base_score)

    get_logger().debug(f'Base score is {minimize_cost}')
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


def remove_worst_windows(valid_emg: list, emg_windows: list, channels=None, weights=None, remove: int = 10) -> [int]:
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    for rem in range(0, remove):

        get_logger().debug(f'Iteration {rem} - Amount of valid windows {len(valid_emg)}')
        idx_ws = 0
        worst_sample = 0
        for sample in range(0, len(valid_emg)):
            minimize_array = []
            for chan in range(0, len(channels)):
                center = int(len(emg_windows[valid_emg[sample]].filtered_data[chan]) / 2)
                minimize_array.append(
                    abs(emg_windows[valid_emg[sample]].filtered_data[chan].idxmin() - center) * weights[chan])

            if sum(minimize_array) > worst_sample:
                worst_sample = sum(minimize_array)
                idx_ws = sample
                get_logger().debug(
                    f'Worst sample: {valid_emg[idx_ws]} - with score: {worst_sample} - based on channels {channels}')

        try:
            del valid_emg[idx_ws]
        except:
            get_logger().exception(f'It was not possible to create a better subset of values. {rem} '
                                   f'values were removed from the array')
            return valid_emg

    get_logger().debug(f'Resulting array: {valid_emg}')
    return valid_emg


def prune_poor_quality_samples(windows: [Window], trigger_table: pd.DataFrame, config,
                               remove: int = 8, method=None):
    # Find valid emgs based on heuristic and calculate averages
    mrcp_windows = 0
    for window in windows:
        if window.label == 1:
            mrcp_windows += 1

    valid_emg = windows_based_on_heuristic(trigger_table, config)  # Our simple time heuristic

    remove = remove - (mrcp_windows - len(valid_emg))
    if remove < 0:
        get_logger().error('You are trying to remove more windows than exists')
        return

    if method:
        valid_emg = method(valid_emg, windows, remove=remove)

    # deletes windows from last index first, in order to avoid index collision
    for i in range(mrcp_windows - 1, -1, -1):
        if i not in valid_emg:
            del windows[i]


def remove_windows_with_blink(dataset, windows):
    blinks = blink_detection(data=dataset.data_device1, sample_rate=dataset.sample_rate)

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
