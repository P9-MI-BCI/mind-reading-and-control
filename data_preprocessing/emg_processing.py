import biosppy
import numpy as np
import pandas as pd
from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter
import matplotlib.pyplot as plt
from scipy.stats import iqr

def emg_amplitude_tkeo(filtered_data):

    tkeo = np.zeros((len(filtered_data),))

    for i in range(1, len(tkeo) - 1):
        tkeo[i] = (filtered_data[i] * filtered_data[i] - filtered_data[i - 1] * filtered_data[i + 1])

    return tkeo


def onset_detection(dataset: Dataset, config, is_online=False) -> [[int]]:
    # Filter EMG Data with specified butterworth filter params from config
    filtered_data = pd.DataFrame()

    # highpass filter to determine onsets
    filtered_data[config.EMG_CHANNEL] = butter_filter(data=dataset.data[config.EMG_CHANNEL],
                                                      order=6,
                                                      cutoff=[30, 300],
                                                      btype='bandpass',
                                                      )

    tkeo = emg_amplitude_tkeo(filtered_data[config.EMG_CHANNEL].to_numpy())

    tkeo_rectified = np.abs(tkeo)

    filtered_data[config.EMG_CHANNEL] = butter_filter(data=tkeo_rectified,
                                                      order=2,
                                                      cutoff=50,
                                                      btype='lowpass',
                                                      )

    # Find onsets based on the filtered data
    onsets, threshold = biosppy.signals.emg.find_onsets(signal=filtered_data[config.EMG_CHANNEL].to_numpy(),
                                                        sampling_rate=dataset.sample_rate,
                                                        )

    t = [threshold] * len(filtered_data[config.EMG_CHANNEL])
    emg_rectified = np.abs(filtered_data[config.EMG_CHANNEL]) > threshold
    emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
    # Group onsets based on time
    emg_clusters = emg_clustering(emg_data=np.abs(filtered_data[config.EMG_CHANNEL]), onsets=emg_onsets, is_online=is_online)

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


def emg_clustering(emg_data, onsets: [int], distance=None, is_online=False) -> [[int]]:
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
            break
        else:
            prev_cluster = clusters

        distance += 200

    for onset_cluster in clusters:
        highest = 0
        index = 0
        for onset in range(onset_cluster[0], onset_cluster[-1] + 1):
            if abs(emg_data[onset]) > highest:
                highest = abs(emg_data[onset])
                index = onset
        all_peaks.append([onset_cluster[0], index, onset_cluster[-1]])

    if is_online:
        return all_peaks
    else:
        return remove_outliers_by_peak_activity(all_peaks, emg_data)


# Compare all peaks and remove outliers below Q1
# We don't care about outliers above Q3 as they have shown clear excess in force
def remove_outliers_by_peak_activity(clusters, emg_data):
    peaks = []
    for cluster in clusters:
        peaks.append(emg_data[cluster[1]])

    iqr_val = iqr(peaks, axis=0)
    Q1 = np.quantile(peaks, 0.25)
    Q3 = np.quantile(peaks, 0.75)
    t_clusters = []

    for cluster in clusters:
        if Q1 - iqr_val*0.7 < emg_data[cluster[1]]:
            t_clusters.append(cluster)

    return t_clusters


def multi_dataset_onset_detection(datasets, config, is_online=False):
    for dataset in datasets:
        dataset.filtered_data = onset_detection(dataset, config, is_online)
