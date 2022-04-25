import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering
from data_visualization import mne_visualization
from classes.Dataset import Dataset


def run(config):
    # subject_id = int(input("Choose subject to predict on 0-9\n"))

    for subject_id in range(10):

        training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)
        training_data = create_dataset(training_dataset_path, config)

        multi_dataset_onset_detection(training_data, config)
        multi_dataset_filtering(config.DELTA_BAND, config, training_data)

        averaged_brain_activity(training_data, config)
    # mne_visualization.visualize_mne(training_data, config)


def averaged_brain_activity(datasets: [Dataset], config):
    channels = config.EEG_CHANNELS
    window_sz = 1200

    cluster_means = []
    for channel in channels:
        cluster_df = pd.DataFrame()
        for dataset in datasets:

            for i, cluster in enumerate(dataset.clusters):
                center = cluster.start
                cluster_win = dataset.filtered_data[channel].iloc[center - window_sz: center + window_sz]

                cluster_win.reset_index(drop=True, inplace=True)
                cluster_df = cluster_df.append(cluster_win.T)

        cluster_mean = cluster_df.mean()
        cluster_means.append(cluster_mean)

    plt.figure()
    plt.pcolor(cluster_means)
    plt.show()

    return cluster_means



def visualize_brain_activity(datasets: [Dataset], config):
    channels = config.EEG_CHANNELS
    window_sz = 1200

    for dataset in datasets:
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(dataset.filename)

        for i, cluster in enumerate(dataset.clusters):
            assert iter(dataset.clusters)
            ax = plt.subplot(6, 4, i + 1)

            center = cluster.start
            data = dataset.filtered_data.iloc[center - window_sz: center + window_sz].iloc[:, :9].T
            im = plt.pcolor(data)

            ax.pcolor(data)
            ax.set_yticks(range(len(channels)))
            ax.set_yticklabels(channels)
            ax.set_title(f'Onset {i}')
            # fig.subplots_adjust(wspace=0, hspace=0)
            ax.axvline(config.SAMPLE_RATE, ls='--', color='black')

            # Onsets
            # onsets = plt.subplot(gs[1], sharex=heatmap)

            # Color bar
            cbar = fig.colorbar(ax=ax, mappable=im, )
            cbar.set_label('Amplitude, uV')

        fig.tight_layout()
        plt.show()
