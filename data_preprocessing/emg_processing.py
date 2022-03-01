from math import floor, ceil
import biosppy
import numpy as np
import pandas as pd
from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter
import matplotlib.pyplot as plt
from scipy.stats import iqr
from utility.logger import get_logger


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
    emg_clusters = emg_clustering(emg_data=np.abs(filtered_data[config.EMG_CHANNEL]), onsets=emg_onsets)

    plot_arr = []
    for cluster in emg_clusters:
        plot_arr.append(filtered_data[config.EMG_CHANNEL].iloc[cluster[0]:cluster[-1]])

    plt.plot(np.abs(filtered_data[config.EMG_CHANNEL]), color='black')
    for vals in plot_arr:
        plt.plot(np.abs(vals))

    if get_logger().level == 10:
        plt.xlabel('Time (s)')
        # plt.xticks([0, 60000, 120000, 180000, 240000, 300000], [0, 50, 100, 150, 200, 250])
        plt.ylabel('mV (Filtered)', labelpad=-2)
        plt.plot(t, '--', color='black')
        plt.autoscale()
        plt.show()

    dataset.onsets_index = emg_clusters
    return filtered_data[config.EMG_CHANNEL]


def emg_clustering(emg_data, onsets: [int], distance=None) -> [[int]]:
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
    return remove_outliers_by_x_axis_distance(all_peaks)
    #return remove_outliers_by_peak_activity(all_peaks, emg_data)


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
    print(len(t_clusters))
    return t_clusters


# Removes any emg clusters that are if < 5*fs or < x*mean/median from the next cluster
# TODO: Decrease naivness, right now it only looks at the cluster ahead of the nth cluster, without any regard for
#       the previous one, and also in a iterative fashion starting from the first cluster.
def remove_outliers_by_x_axis_distance(clusters):
    t_clusters = []
    temp = 0
    temp_arr = []
    x = 0.4

    # Find distance between emg cluster peaks and calculate mean
    for i in range(0, len(clusters)-1):
        temp_arr.append(abs(clusters[i][1] - clusters[i+1][1]))
        temp = temp + abs(clusters[i][1] - clusters[i+1][1])
    mean = temp/len(clusters)
    temp_arr.sort()

    # Get median of distance between emg cluster peaks
    if len(temp_arr) % 2 == 0:
        median = temp_arr[floor((len(temp_arr)-1)/2)] + temp_arr[ceil((len(temp_arr)-1)/2)] / 2
    else:
        median = temp_arr[(len(temp_arr)-1)//2]

    # Include only clusters which peaks are 5*sample_rate and x*mean/median frequencies apart
    for i in range(0, len(clusters)-1):
        if abs(clusters[i][1] - clusters[i+1][1]) > 5*1200 and abs(clusters[i][1] - clusters[i+1][1]) > x*median: # TODO: Change 1200 to dataset fs
            t_clusters.append(clusters[i])

    # Handle 'edge' case for last element of array
    if abs(clusters[-1][1] - clusters[-2][1]) > 5*1200 and abs(clusters[-1][1] - clusters[-2][1]) > x*median:
        t_clusters.append(clusters[-1])

    print(len(t_clusters))
    return t_clusters

# TODO: 2d outlier detection (width of cluster, height of cluster)
#       Signal to noise ratio.

def multi_dataset_onset_detection(datasets, config):
    for dataset in datasets:
        dataset.filtered_data = onset_detection(dataset, config)
