# performs a grid search to cost minimize array.
import json

from data_preprocessing.mrcp_detection import mrcp_detection
from data_visualization.average_channels import find_usable_emg, average_channel
from utility.logger import get_logger
from tqdm import tqdm


# test distance from minimum to middle
# what sample has the biggest negative influence on the average

def find_best_config_params(data, trigger_table, config):
    emg_order_range = [4, 5, 6]
    emg_cutoff_range = list(range(75, 110))
    eeg_cutoff_range_min = [0.04, 0.05, 0.06, 0.07]
    eeg_cutoff_range_max = [4, 5, 6, 7, 8, 9, 10]
    eeg_order = [2,3]

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
                            emg_windows, trigger_table = mrcp_detection(data=data, tp_table=trigger_table, config=config)

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
                                get_logger().info(f'New shortest distance/cost {minimize_cost}')
                                get_logger().info(f'config: {config}')
                        except:
                            get_logger().debug(f'During param search - Config : {config} did not work.')
        # for some reason always prints the last params ..
    print(minimized_config)


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

    get_logger().info(f'Base score is {minimize_cost}')
    try:
        for rem in range(0, remove):
            worst_sample = None
            get_logger().info(f'Current Valid EMGs {valid_emg}')

            for sample in range(0, len(valid_emg)):
                minimize_array = []

                b = [x for i, x in enumerate(valid_emg) if i != sample]
                windows = average_channel(emg_windows, b)

                for i in range(0, len(windows)):
                    if i in channels:
                        minimize_array.append(abs(windows[i].data.idxmin() - int(len(windows[i].data) / 2)) * weights[i])

                if sum(minimize_array) <= minimize_cost:
                    worst_sample = sample
                    minimize_cost = sum(minimize_array)
                    get_logger().info(f'New shortest distance/cost {minimize_cost}')
                    get_logger().info(f'Attained by removing sample with index {worst_sample}')

            try:
                del valid_emg[worst_sample]
            except TypeError:
                get_logger().info(f'It was not possible to create a better subset of values. {rem} '
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

        get_logger().info(f'Iteration {rem} - Amount of valid windows {len(valid_emg)}')
        idx_ws = 0
        worst_sample = 0
        for sample in range(0, len(valid_emg)):
            minimize_array = []
            for chan in range(0, len(channels)):
                center = int(len(emg_windows[valid_emg[sample]].filtered_data[channels[chan]]) / 2)
                minimize_array.append(abs(emg_windows[valid_emg[sample]].filtered_data[channels[chan]].idxmin() - center) * weights[channels[chan]])

            if sum(minimize_array) > worst_sample:
                worst_sample = sum(minimize_array)
                idx_ws = sample
                get_logger().info(
                    f'Worst sample: {valid_emg[idx_ws]} - with score: {worst_sample} - based on channels {channels}')

        try:
            del valid_emg[idx_ws]
        except:
            get_logger().info(f'It was not possible to create a better subset of values. {rem} '
                              f'values were removed from the array')
            return valid_emg

    get_logger().info(f'Resulting array: {valid_emg}')
    return valid_emg
