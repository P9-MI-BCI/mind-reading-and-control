import copy

import matplotlib.pyplot as plt

from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.filters import multi_dataset_filtering


def run(config):
    subject_id = 1

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    raw_data = create_dataset(training_dataset_path, config)
    notch_data = copy.deepcopy(raw_data)
    bandpass_data = copy.deepcopy(raw_data)
    notch_bandpass_data = copy.deepcopy(raw_data)

    multi_dataset_filtering(config.NOTCH, config, notch_data, notch=True)

    multi_dataset_filtering(config.BASELINE, config, bandpass_data, notch=False)

    multi_dataset_filtering(config.NOTCH, config, notch_bandpass_data, notch=True)
    multi_dataset_filtering(config.BASELINE, config, notch_bandpass_data, notch=False)

    visualize_signals(raw_data, notch_data, bandpass_data, notch_bandpass_data)


def visualize_signals(raw_data, notch_data, bandpass_data, notch_bandpass_data):
    fig, axs = plt.subplots(4)
    channel = 'Cz'
    range = 12000

    axs[0].plot(raw_data[0].data[channel].iloc[:range])
    axs[0].set_title('Raw signal')

    axs[1].plot(notch_data[0].filtered_data[channel].iloc[:range])
    axs[1].set_title('Notch filtered signal')

    axs[2].plot(bandpass_data[0].filtered_data[channel].iloc[:range])
    axs[2].set_title('Bandpass filtered signal')

    axs[3].plot(notch_bandpass_data[0].filtered_data[channel].iloc[:range])
    axs[3].set_title('Notch + bandpass filtered signal')

    plt.tight_layout()
    plt.show()



