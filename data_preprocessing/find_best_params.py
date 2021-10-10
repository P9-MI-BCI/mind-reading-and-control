# performs a grid search to cost minimize array.
import json

from data_preprocessing.mrcp_detection import mrcp_detection
from data_visualization.average_channels import find_usable_emg, average_channel
from utility.logger import get_logger
from tqdm import tqdm

# test distance from minimum to middle
# what sample has the biggest negative influence on the average
def find_best_params(data, trigger_table, config):
    # with open('config.json', 'w') as config_file:
    #     json.dump(config, config_file)

    emg_order_range = [5,4]
    emg_cutoff_range = list(range(75,100))
    eeg_cutoff_range_min = [0.03, 0.04, 0.05]
    eeg_cutoff_range_max = [3, 4, 5]

    channels = [0, 1, 3, 4, 6, 7, 8]

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
                        valid_emg = find_usable_emg(trigger_table, config)
                        frames = average_channel(emg_frames, valid_emg)

                        minimize_array = []
                        for i in range(0, len(frames)):
                            if i in channels:
                                minimize_array.append(abs(frames[i].data.idxmin() - int(len(frames[i].data)/2)))
                            # distance = frame.data.min().index - frame.data.
                            # if int(len(frame.data)/2) in frame.data.nsmallest(250):
                            #     minimize_array.append(1)
                            # else:
                            #     minimize_array.append(0)

                        if sum(minimize_array) <= minimize_cost:
                            minimized_config = config
                            minimize_cost = sum(minimize_array)
                            print(minimize_array)
                    except ValueError:
                        get_logger().debug(f'During param search - Config : {config} did not work.')
    # for some reason always prints the last params ..
    print(minimized_config)


def find_worst_sample(valid_emg, emg_frames, remove: int = 10):
    channels = [0, 1, 3, 4, 6, 7, 8]

    base_frames = average_channel(emg_frames, valid_emg)

    base_score = []
    for i in range(0, len(base_frames)):
        if i in channels:
            base_score.append(abs(base_frames[i].data.idxmin() - int(len(base_frames[i].data) / 2)))

    minimize_cost = sum(base_score)

    get_logger().info(f'Base score is {minimize_cost}')

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

    return valid_emg
