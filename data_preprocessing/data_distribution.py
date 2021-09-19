import collections
import random

from data_preprocessing.trigger_points import is_triggered
from classes import Frame


def aggregate_data(device_data_pd, freq_size, is_triggered_table, sample_rate=1200):
    list_of_dataframes = []

    for i in range(0, device_data_pd.shape[0], freq_size):
        # the label for the frame is attached first. we base being 'triggered' whether the middle frequency is
        # recorded during the triggered timeframe.
        data_frame = Frame.Frame()

        data_frame.label = is_triggered(i + freq_size / 2, is_triggered_table, sample_rate)
        data_frame.data = device_data_pd.iloc[i:i + freq_size]

        list_of_dataframes.append(data_frame)

    # return all but the last frame, because it is not complete
    return list_of_dataframes[:-1]


# todo generalize for x features
def data_distribution(labelled_data_lst):
    triggered = 0

    for frame in labelled_data_lst:
        if frame.label == 1:
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
        features.append(frame.label)

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
        if feat_counter[frame.label] < least_represented_feature:
            uniform_data_list.append(frame)
            feat_counter[frame.label] += 1

    return uniform_data_list


def z_score_normalization(data_list):
    for frame in data_list:
        frame.data = (frame.data - frame.data.mean()) / frame.data.std()

    return data_list

