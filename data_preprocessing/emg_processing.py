from math import floor, ceil
import biosppy
import numpy as np
import pandas as pd
from classes.Dataset import Dataset
from classes.Cluster import Cluster
from data_preprocessing.filters import butter_filter
import matplotlib.pyplot as plt
from scipy.stats import iqr
from utility.logger import get_logger


def emg_amplitude_tkeo(filtered_data):
    tkeo = np.zeros((len(filtered_data),))

    for i in range(1, len(tkeo) - 1):
        tkeo[i] = (filtered_data[i] * filtered_data[i] - filtered_data[i - 1] * filtered_data[i + 1])

    return tkeo


def onset_detection(dataset: Dataset, config, is_online=False, prox_coef=2) -> [[int]]:
    # Filter EMG Data with specified butterworth filter params from config
    filtered_data = pd.DataFrame()

    bandpass_data = butter_filter(data=dataset.data[config.EMG_CHANNEL],
                                  order=config.EMG_ORDER_BANDPASS,
                                  cutoff=config.EMG_CUTOFF_BANDPASS,
                                  btype=config.EMG_BTYPE_BANDPASS,
                                  )

    tkeo = emg_amplitude_tkeo(bandpass_data)

    tkeo_rectified = np.abs(tkeo)

    filtered_data[config.EMG_CHANNEL] = butter_filter(data=tkeo_rectified,
                                                      order=config.EMG_ORDER_LOWPASS,
                                                      cutoff=config.EMG_CUTOFF_LOWPASS,
                                                      btype=config.EMG_BTYPE_LOWPASS,
                                                      )

    # Find onsets based on the filtered data
    onsets, threshold = biosppy.signals.emg.find_onsets(signal=filtered_data[config.EMG_CHANNEL].to_numpy(),
                                                        sampling_rate=dataset.sample_rate,
                                                        )
    emg_rectified = np.abs(filtered_data[config.EMG_CHANNEL]) > threshold
    emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()

    # Group onsets based on time
    emg_clusters, filtered_data[config.EMG_CHANNEL], threshold = emg_clustering(
        emg_data=np.abs(filtered_data[config.EMG_CHANNEL]), onsets=emg_onsets,
        is_online=is_online, prox_coef=prox_coef)

    # Plotting of EMG signal and clusters
    t = [threshold] * len(filtered_data[config.EMG_CHANNEL])

    if get_logger().level == 10:
        cluster_plot_arr = []
        for cluster in emg_clusters:
            cluster_plot_arr.append(filtered_data[config.EMG_CHANNEL].iloc[cluster.start:cluster.end])

        plt.plot(np.abs(filtered_data[config.EMG_CHANNEL]), color='black')

        for vals in cluster_plot_arr:
            plt.plot(np.abs(vals))

        plt.plot(t, '--', color='black')
        plt.title(dataset.filename)
        plt.xlabel('Time (s)')
        plt.ylabel('mV (Filtered)', labelpad=-2)
        plt.autoscale()
        plt.show()

    dataset.onsets_index = emg_clusters
    return filtered_data[config.EMG_CHANNEL]


def emg_clustering(emg_data, onsets: [int], distance=None, is_online=False, prox_coef=2, normalize=True,
                   threshold=None) -> [[int]]:
    # TODO: Fix clustering to detect fixed amount of clusters
    all_peaks = []
    if distance is None:
        distance = 100

    prev_cluster = []

    while True:
        clusters = []
        temp_clusters = []
        cluster_list = []
        for idx in onsets:
            if len(temp_clusters) == 0:
                temp_clusters.append(idx)
            elif abs(temp_clusters[-1] - idx) < distance:
                temp_clusters.append(idx)
            else:
                clusters.append(temp_clusters)
                temp_clusterobj = Cluster(data=temp_clusters)
                temp_clusterobj.create_info()
                cluster_list.append(temp_clusterobj)
                temp_clusters = [idx]

        clusters.append(temp_clusters)
        temp_clusterobj = Cluster(data=temp_clusters)
        temp_clusterobj.create_info()
        cluster_list.append(temp_clusterobj)

        if clusters == prev_cluster:
            break
        else:
            prev_cluster = clusters

        distance += 200

    try:
        assert len(clusters) > 2
        cluster_list = remove_outliers_by_x_axis_distance(cluster_list, prox_coef)

        for onset_cluster in cluster_list:
            highest = 0
            index = 0
            for onset in range(onset_cluster.data[0], onset_cluster.data[-1]):
                if abs(emg_data[onset]) > highest:
                    highest = abs(emg_data[onset])
                    index = onset
            onset_cluster.peak = index

        if (normalize):
            onsets, threshold, emg_data = normalize_peaks(cluster_list, np.abs(emg_data))
            emg_rectified = np.abs(emg_data) > threshold
            emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
            cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data), onsets=emg_onsets,
                                                               is_online=is_online, prox_coef=prox_coef,
                                                               normalize=False, threshold=threshold)

        if is_online:
            return cluster_list, emg_data, threshold
        else:
            # cluster_list, emg_data = remove_outliers_by_peak_activity(cluster_list, emg_data)
            return cluster_list, emg_data, threshold
    except AssertionError:
        get_logger().exception(f'File only contains {len(cluster_list)} clusters.')


def normalize_peaks(clusters, emg_data):
    # TODO: Consider hardcap of how high cluster can be
    peaks = []
    for cluster in clusters:
        peaks.append(emg_data[cluster.peak])

    iqr_val = iqr(peaks, axis=0)
    Q1 = np.quantile(peaks, 0.25)
    Q3 = np.quantile(peaks, 0.75)

    for cluster in clusters:
        if Q3 < emg_data[cluster.peak]:
            for index in cluster.data:
                if Q3 < emg_data[index]:
                    if Q3 < emg_data[index] / 2:
                        emg_data[index] = emg_data[index] / 4
                    else:
                        emg_data[index] = emg_data[index] / 2

        if Q1 > emg_data[cluster.peak]:
            for index in cluster.data:
                if Q1 > emg_data[index]:
                    emg_data[index] = emg_data[index] + iqr_val

    o, t = biosppy.signals.emg.find_onsets(signal=emg_data.to_numpy(), sampling_rate=1200)

    return o, t, emg_data


# Compare all peaks and remove outliers below Q1
# We don't care about outliers above Q3 as they have shown clear excess in force
def remove_outliers_by_peak_activity(clusters, emg_data):
    # TODO: Normalize values > Q3 by iqr
    peaks = []
    for cluster in clusters:
        peaks.append(emg_data[cluster.peak])

    iqr_val = iqr(peaks, axis=0)
    Q1 = np.quantile(peaks, 0.25)
    Q3 = np.quantile(peaks, 0.75)

    t_clusters = []

    for cluster in clusters:
        if Q1 - iqr_val * 0.7 < emg_data[cluster.peak]:
            t_clusters.append(cluster)

    return t_clusters, emg_data


def remove_outliers_by_x_axis_distance(clusters, prox_coef):
    # TODO: Figure out how to incorporate in EMG clustering after changed to detect fixed 20 clusters
    clusters_to_remove = []
    t_clusters = []

    for i in range(0, len(clusters) - 2):
        # Check for all clusters if the subsequent cluster is closer in proximity than x*fs
        if abs(clusters[i].end - clusters[i + 1].start) < prox_coef * 1200:
            # Check which one of the clusters are the largest (naive way of selecting which one is cluster and which one is outlier)
            if len(clusters[i].data) < len(clusters[i + 1].data):
                clusters[i + 1].data = np.append(clusters[i + 1].data, clusters[i].data)
                clusters[i + 1].create_info()
                clusters_to_remove.append(clusters[i])
                i = i + 1
            else:
                clusters[i].data = np.append(clusters[i].data, clusters[i + 1].data)
                clusters[i].create_info()
                clusters_to_remove.append(clusters[i + 1])

    # Handle 'edge' case for last element of array
    if abs(clusters[-1].start - clusters[-2].end) < prox_coef * 1200:
        if len(clusters[-1].data) < len(clusters[-2].data):
            clusters[-2].data = np.append(clusters[-2].data, clusters[-1].data)
            clusters[-2].create_info()
            clusters_to_remove.append(clusters[-1])
        else:
            clusters[-1].data = np.append(clusters[-1].data, clusters[-2].data)
            clusters[-1].create_info()
            clusters_to_remove.append(clusters[-2])

    if (clusters_to_remove):
        get_logger().debug(f"Removed {len(clusters_to_remove)} clusters because of proximity")
    for cluster in clusters:
        if cluster not in clusters_to_remove:
            t_clusters.append(cluster)

    return t_clusters


def multi_dataset_onset_detection(datasets, config, is_online=False):
    try:
        iter(datasets)
    except TypeError:
        datasets.filtered_data = onset_detection(datasets, config, is_online)
        datasets.data = datasets.data.iloc[1:-1, :]
        datasets.data.reset_index(drop=True, inplace=True)
    else:
        for dataset in datasets:
            dataset.filtered_data = onset_detection(dataset, config, is_online)
            dataset.data = dataset.data.iloc[1:-1, :]
            dataset.data.reset_index(drop=True, inplace=True)
