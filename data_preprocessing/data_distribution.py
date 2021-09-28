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


# finds the start of trigger point and converts it to frequency and takes the frame_size (in seconds) and cuts each
# side into a dataframe.
# this is used to find peaks locally in EMG data.
def aggregate_trigger_points_for_emg_peak(tp_table, column, data, frame_size=2):
    list_of_trigger_frames = []
    indices_to_delete = []

    for i, row in tp_table.iterrows():
        start = int(row[column].total_seconds() * data.sample_rate - frame_size * data.sample_rate)
        end = int(row[column].total_seconds() * data.sample_rate + frame_size * data.sample_rate)
        frame = Frame.Frame()
        frame.data = data.data_device1.iloc[start:end]
        frame.label = 1  # indicates EMG peak
        indices_to_delete.append([start, end])
        list_of_trigger_frames.append(frame)

    indices_to_delete.reverse()

    for indices in indices_to_delete:
        data.data_device1 = data.data_device1.drop(data.data_device1.index[indices[0]:indices[1]])

    return list_of_trigger_frames, data


def slice_and_label_idle_frames(data, frame_size=4800):
    list_of_frames = []
    i = 0
    while i < len(data) and i + frame_size < len(data):
        cutout = abs(data.index[i] - data.index[i+frame_size]) == frame_size
        if cutout:
            frame = Frame.Frame()
            frame.data = data.iloc[i:i+frame_size]
            frame.label = 0  # indicates no EMG peak / no MRCP should be present
            list_of_frames.append(frame)
            i += frame_size
        else:
            i += 1

    return list_of_frames




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


def z_score_normalization(frame):
    return (frame - frame.mean()) / frame.std()


def max_absolute_scaling(frame):
    return frame / frame.abs().max()


def min_max_scaling(frame):
    return (frame - frame.min()) / (frame.max() - frame.min())
