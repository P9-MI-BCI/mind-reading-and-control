import collections
import random

from data_preprocessing.trigger_points import is_triggered


def aggregate_data(device_data_pd, freq_size, is_triggered_table):
    list_of_dataframes = []
    counter = 0

    for i in range(0, device_data_pd.shape[0], freq_size):
        # the label for the frame is attached first. we base being 'triggered' whether the center frequency is recorded during the triggered timeframe.
        list_of_dataframes.append(
            [is_triggered(i + freq_size / 2, is_triggered_table), device_data_pd.iloc[i:i + freq_size]])

        # notebook cant handle all the data at once.

    return list_of_dataframes


def data_distribution(labelled_data_lst):
    triggered = 0

    for frame in labelled_data_lst:
        if frame[0] == 1:
            triggered += 1
    # todo  counter = collections.Counter(features)

    idle = len(labelled_data_lst) - triggered
    return {
        'triggered': triggered,
        'idle': idle,
        'expected_triggered_percent': int(triggered / (triggered + idle) * 100)
    }


def create_uniform_distribution(data_list):
    # returns the dataset with equal amount of samples, chosen by the least represented feature.
    features = []

    for frame in data_list:
        features.append(frame[0])

    counter = collections.Counter(features)
    least_represented_feature = 99999999999
    feat_counter = {}
    for i in counter.keys():
        feat_counter[i] = 0
        if counter[i] < least_represented_feature:
            least_represented_feature = counter[i]

    random.shuffle(data_list)

    uniform_data_list = []
    for frame in data_list:
        if feat_counter[frame[0]] < least_represented_feature:
            uniform_data_list.append(frame)
            feat_counter[frame[0]] += 1

    return uniform_data_list


def z_score_normalization(data_list):
    z_scored_data = []
    for frame in data_list:
        _temp = (frame[1] - frame[1].mean()) / frame[1].std()
        z_scored_data.append([frame[0], _temp])

    return z_scored_data

