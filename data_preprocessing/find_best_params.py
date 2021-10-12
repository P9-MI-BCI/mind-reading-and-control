# performs a grid search to cost minimize array.
import json

from data_preprocessing.mrcp_detection import mrcp_detection
from data_visualization.average_channels import find_usable_emg, average_channel
from utility.logger import get_logger
from tqdm import tqdm


# test distance from minimum to middle
# what sample has the biggest negative influence on the average

def find_best_config_params(data, trigger_table, config):
    emg_order_range = [4, 5]
    emg_cutoff_range = list(range(75, 110))
    eeg_cutoff_range_min = [0.03, 0.04, 0.05]
    eeg_cutoff_range_max = [3, 4, 5]

    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    minimize_cost = 99999999
    minimized_config = {}

    for order in tqdm(emg_order_range):
        for cutoff in tqdm(emg_cutoff_range):
            for eeg_min in eeg_cutoff_range_min:
                for eeg_max in eeg_cutoff_range_max:
                    config['emg_order'] = order
                    config['emg_cutoff'] = cutoff
                    config['eeg_cutoff'] = [eeg_min, eeg_max]

                    try:
                        emg_frames, trigger_table = mrcp_detection(data=data, tp_table=trigger_table, config=config)

                        # Find valid emgs based on heuristic and calculate averages
                        valid_emg = [3, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
                        frames = average_channel(emg_frames, valid_emg)

                        minimize_array = []
                        for i in range(0, len(frames)):
                            if i in channels:
                                minimize_array.append(abs(frames[i].data.idxmin() - int(len(frames[i].data) / 2)))

                        if sum(minimize_array) <= minimize_cost:
                            minimized_config = config
                            minimize_cost = sum(minimize_array)
                            print(f' Score: {sum(minimize_array)}')
                            print(f'Config: {config}')
                    except ValueError:
                        get_logger().debug(f'During param search - Config : {config} did not work.')
    # for some reason always prints the last params ..
    print(minimized_config)


def optimize_average_minimum(valid_emg, emg_frames, channels=None, remove: int = 10):
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    base_frames = average_channel(emg_frames, valid_emg)

    base_score = []
    for i in range(0, len(base_frames)):
        if i in channels:
            base_score.append(abs(base_frames[i].data.idxmin() - int(len(base_frames[i].data) / 2)))

    minimize_cost = sum(base_score)

    get_logger().info(f'Base score is {minimize_cost}')
    try:
        for rem in range(0, remove):
            get_logger().info(f'Current Valid EMGs {valid_emg}')

            for sample in range(0, len(valid_emg)):
                minimize_array = []

                b = [x for i, x in enumerate(valid_emg) if i != sample]
                frames = average_channel(emg_frames, b)

                for i in range(0, len(frames)):
                    if i in channels:
                        minimize_array.append(abs(frames[i].data.idxmin() - int(len(frames[i].data) / 2)))

                if sum(minimize_array) <= minimize_cost:
                    worst_sample = sample
                    minimize_cost = sum(minimize_array)
                    get_logger().info(f'New shortest distance/cost {minimize_cost}')
                    get_logger().info(f'Attained by removing sample with index {worst_sample}')

            del valid_emg[worst_sample]
    except ValueError:
        get_logger().error(f'You are trying to remove more samples than there is valid emgs detected. There are '
                           f'{len(valid_emg)} valid emgs you are trying to remove {rem} more.')

    return valid_emg


def remove_worst_frames(valid_emg: list, emg_frames: list, channels=None, remove: int = 10) -> [int]:
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for rem in range(0, remove):

        get_logger().info(f'Iteration {rem} - Amount of valid frames {len(valid_emg)}')
        idx_ws = 0
        worst_sample = 0
        for sample in range(0, len(valid_emg)):
            minimize_array = []
            for chan in channels:
                center = int(len(emg_frames[valid_emg[sample]].filtered_data[chan]) / 2)
                minimize_array.append(abs(emg_frames[valid_emg[sample]].filtered_data[chan].idxmin() - center))

            if sum(minimize_array) > worst_sample:
                worst_sample = sum(minimize_array)
                idx_ws = sample
                get_logger().info(
                    f'Worst sample: {valid_emg[idx_ws]} - with score: {worst_sample} - based on channels {channels}')

        del valid_emg[idx_ws]

    get_logger().info(f'Resulting array: {valid_emg}')
    return valid_emg
