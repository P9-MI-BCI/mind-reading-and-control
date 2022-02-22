import biosppy
import numpy as np
import pandas as pd
from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter
import matplotlib.pyplot as plt
from scipy.stats import iqr


def onset_detection(dataset: Dataset, config) -> [[int]]:
    # Filter EMG Data with specified butterworth filter params from config
    filtered_data = pd.DataFrame()

    # highpass filter to determine onsets
    filtered_data[config.EMG_CHANNEL] = butter_filter(data=dataset.data[config.EMG_CHANNEL],
                                                      order=config.EMG_ORDER,
                                                      cutoff=config.EMG_CUTOFF,
                                                      btype=config.EMG_BTYPE,
                                                      )

    # Find onsets based on the filtered data
    onsets, threshold = biosppy.signals.emg.find_onsets(signal=filtered_data[config.EMG_CHANNEL].to_numpy(),
                                                        sampling_rate=dataset.sample_rate,
                                                        )

    t = [threshold] * len(filtered_data[config.EMG_CHANNEL])
    emg_rectified = np.abs(filtered_data[config.EMG_CHANNEL]) > threshold
    emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
    # Group onsets based on time
    emg_clusters = emg_clustering(onsets=emg_onsets)

    plot_arr = []
    for cluster in emg_clusters:
        plot_arr.append(filtered_data[config.EMG_CHANNEL].iloc[cluster[0]:cluster[-1]])

    plt.plot(np.abs(filtered_data[config.EMG_CHANNEL]), color='black')
    for vals in plot_arr:
        plt.plot(np.abs(vals))

    plt.xlabel('Time (s)')
    # plt.xticks([0, 60000, 120000, 180000, 240000, 300000], [0, 50, 100, 150, 200, 250])
    plt.ylabel('mV (Filtered)', labelpad=-2)
    plt.plot(t, '--', color='black')
    plt.autoscale()
    plt.show()

    dataset.onsets_index = emg_clusters
    return filtered_data[config.EMG_CHANNEL]


def emg_clustering(onsets: [int], distance=None) -> [[int]]:
    all_peaks = []
    if distance is None:
        distance = 100

    prev_cluster = []

    while True:
        clusters = []
        temp_clusters = []
        for idx in onsets:
            if len(temp_clusters) == 0:
                temp_clusters.append(idx)
            elif abs(temp_clusters[-1] - idx) < distance:
                temp_clusters.append(idx)
            else:
                clusters.append(temp_clusters)
                temp_clusters = [idx]
        clusters.append(temp_clusters)

        if clusters == prev_cluster:

            lengths = [len(x) for x in clusters]
            iqr_val = iqr(lengths, axis=0)
            Q1 = np.quantile(lengths, 0.25)
            Q3 = np.quantile(lengths, 0.75)
            t_clusters = []

            for cluster in clusters:
                if Q1 - iqr_val < len(cluster):
                    t_clusters.append(cluster)

            clusters = t_clusters
            break
        else:
            prev_cluster = clusters

        distance += 200

    for onset_cluster in clusters:
        all_peaks.append([onset_cluster[0], onset_cluster[-1]])

    return all_peaks


def multi_dataset_onset_detection(datasets, config):
    for dataset in datasets:
        dataset.filtered_data = onset_detection(dataset, config)
