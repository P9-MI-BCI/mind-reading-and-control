import copy

import matplotlib.pyplot as plt

from data_preprocessing.data_distribution import data_preparation, normalization
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.filters import multi_dataset_filtering
from data_preprocessing.emg_processing import onset_detection, multi_dataset_onset_detection


def run(config):
    subject_id = 1

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    raw_data = create_dataset(training_dataset_path, config)

    baseline = copy.deepcopy(raw_data)
    baseline_notch = copy.deepcopy(raw_data)
    raw_delta = copy.deepcopy(raw_data)
    baseline_delta = copy.deepcopy(raw_data)
    baseline_delta_notch = copy.deepcopy(raw_data)

    multi_dataset_filtering(config.BASELINE, config, baseline, notch=False)

    multi_dataset_filtering(config.BASELINE, config, baseline_notch, notch=False)
    multi_dataset_filtering(config.NOTCH, config, baseline_notch, notch=True)

    multi_dataset_filtering(config.DELTA_BAND, config, raw_delta, notch=False)

    multi_dataset_filtering(config.BASELINE, config, baseline_delta, notch=False)
    multi_dataset_filtering(config.DELTA_BAND, config, baseline_delta, notch=False)

    multi_dataset_filtering(config.BASELINE, config, baseline_delta_notch, notch=False)
    multi_dataset_filtering(config.NOTCH, config, baseline_delta_notch, notch=True)
    multi_dataset_filtering(config.DELTA_BAND, config, baseline_delta_notch, notch=False)

    multi_dataset_onset_detection(raw_delta, config)

    X, Y = data_preparation(raw_delta, config)
    X, scaler = normalization(X)
    # onset_detection(raw_data[2], config)

    visualize_all_channels(raw_data, raw_delta, X, config)

    # visualize_signals(baseline, baseline_notch, raw_delta, baseline_delta, baseline_delta_notch, config)


def visualize_all_channels(raw_data, delta, X, config):
    range = 12000

    fig, ax = plt.subplots(9, figsize=(75,5))
    for i, channel in enumerate(config.EEG_CHANNELS):
        ax[i].plot(X[i], 'k')
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_signals(baseline, baseline_notch, raw_delta, baseline_delta, baseline_delta_notch, config):
    channel = 'Cz'
    range = 12000

    fig, axs = plt.subplots(5)

    axs[0].plot(baseline[0].filtered_data[channel].iloc[:range])
    axs[0].set_title('Baseline')

    axs[1].plot(baseline_notch[0].filtered_data[channel].iloc[:range])
    axs[1].set_title('Baseline + notch')

    axs[2].plot(raw_delta[0].filtered_data[channel].iloc[:range])
    axs[2].set_title('Raw + delta')

    axs[3].plot(baseline_delta[0].filtered_data[channel].iloc[:range])
    axs[3].set_title('Baseline + delta')

    axs[4].plot(baseline_delta_notch[0].filtered_data[channel].iloc[:range])
    axs[4].set_title('Baseline + notch + delta')

    plt.tight_layout()
    plt.show()

    list_1 = raw_delta[0].filtered_data[channel].iloc[:range]
    list_2 = baseline_delta[0].filtered_data[channel].iloc[:range]
    difference_1 = set(list_1).difference(set(list_2))
    difference_2 = set(list_2).difference(set(list_1))

    list_difference = list(difference_1.union(difference_2))
    print(list_difference)
