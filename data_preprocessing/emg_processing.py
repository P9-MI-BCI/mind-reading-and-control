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


def onset_detection(dataset: Dataset, config, static_clusters=True,
                    proximity_outliers=True, normalization=True) -> [[int]]:
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
        emg_data=np.abs(filtered_data[config.EMG_CHANNEL]), onsets=emg_onsets, threshold=threshold,
        static_clusters=static_clusters, proximity_outliers=proximity_outliers, normalization=normalization)

    # Plotting of EMG signal and clusters
    t = [threshold] * len(filtered_data[config.EMG_CHANNEL])

    if get_logger().level == 10:
        cluster_plot_arr = []
        for cluster in emg_clusters:
            cluster_plot_arr.append(filtered_data[config.EMG_CHANNEL].iloc[cluster.start:cluster.end])

        fig = plt.figure(figsize=(20, 8))
        plt.plot(np.abs(filtered_data[config.EMG_CHANNEL]), color='black')

        for vals in cluster_plot_arr:
            plt.plot(np.abs(vals))

        plt.plot(t, '--', color='black')
        plt.title(dataset.filename)
        plt.xlabel('Time (s)')
        plt.ylabel('mV (Filtered)', labelpad=-2)
        plt.autoscale()
        plt.show()

    dataset.clusters = emg_clusters
    return filtered_data[config.EMG_CHANNEL]


def emg_clustering(emg_data, onsets: [int], distance=None, normalization=True,
                   threshold=None, static_clusters=True, proximity_outliers=True) -> [
    [int]]:
    if distance is None:
        distance = 100

    prev_cluster = []
    stop_loop = 0
    noise_detected = False

    while True:
        removed_cluster = False
        temp_clusters = []
        cluster_list = []
        # For indices in detected onsets
        for idx in onsets:
            # First element check
            if len(temp_clusters) == 0:
                temp_clusters.append(idx)
            # If the last element in temp array's distance to the next index is less than 'distance', add this index
            # to this cluster
            elif abs(temp_clusters[-1] - idx) < distance:
                temp_clusters.append(idx)
            # Else create cluster object and add index to next cluster list
            else:
                temp_clusterobj = Cluster(data=temp_clusters)
                temp_clusterobj.create_info()
                cluster_list.append(temp_clusterobj)
                temp_clusters = [idx]

        # For last cluster
        temp_clusterobj = Cluster(data=temp_clusters)
        temp_clusterobj.create_info()
        cluster_list.append(temp_clusterobj)

        # Check if no changes in previous iteration and break, else increase distance
        if static_clusters:
            foo = []
            small_clusters = 0
            if len(cluster_list) == 21:
                for cluster in cluster_list:
                    foo.append(len(cluster.data))
                median = np.median(foo)
                # TODO: Try this some more:
                # small_clusters = [small_clusters+1 for size in foo if size < median*0.5]
                for cluster in cluster_list:
                    if len(cluster.data) < median * 0.1:
                        cluster_list.remove(cluster)
                        removed_cluster = True
                        noise_detected = True
                if removed_cluster:
                    break
                else:
                    distance += 20
            if len(cluster_list) == 20:
                break
            elif len(cluster_list) > 20:
                distance += 20
            elif len(cluster_list) < 20 and stop_loop < 20:
                stop_loop += 1
                distance -= 1
                noise_detected = True
            else:
                break
        else:
            if prev_cluster == cluster_list:
                break
            else:
                prev_cluster = cluster_list
                distance += 100


    try:
        # Remove outliers by looking at their proximity
        if proximity_outliers:
            assert len(cluster_list) > 2
            cluster_list = remove_outliers_by_x_axis_distance(cluster_list)

        # Find peaks of clusters
        for onset_cluster in cluster_list:
            highest = 0
            index = 0
            for onset in range(onset_cluster.start, onset_cluster.end):
                if abs(emg_data[onset]) > highest:
                    highest = abs(emg_data[onset])
                    index = onset
            onset_cluster.peak = index

        # Normalize the EMG data a single time
        if normalization:
            onsets, threshold, emg_data = normalize_peaks(cluster_list, np.abs(emg_data))
            if noise_detected:  # or small_clusters:
                threshold = threshold * 0.5
            emg_rectified = np.abs(emg_data) > threshold
            emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
            cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data),
                                                               onsets=emg_onsets,
                                                               normalization=False,
                                                               threshold=threshold,
                                                               static_clusters=static_clusters,
                                                               proximity_outliers=False)

        return cluster_list, emg_data, threshold
    except AssertionError:
        get_logger().exception(f'File only contains {len(cluster_list)} clusters.')


def normalize_peaks(clusters, emg_data):
    peaks = []
    for cluster in clusters:
        peaks.append(emg_data[cluster.peak])

    iqr_val = iqr(peaks, axis=0)
    Q1 = np.quantile(peaks, 0.25)
    Q3 = np.quantile(peaks, 0.75)

    # Half along the y-axis any cluster with indices over Q1 until they are below Q1
    for cluster in clusters:
        if Q1 < emg_data[cluster.peak]:
            for index in cluster.data:
                while Q1 < emg_data[index]:
                    emg_data[index] = emg_data[index] / 2

        # Increase by the IQR Value along the y-axis for the clusters below Q1
        # if Q1 > emg_data[cluster.peak]:
        #     for index in cluster.data:
        #         if Q1 > emg_data[index]:
        #             emg_data[index] = emg_data[index] + iqr_val

    o, t = biosppy.signals.emg.find_onsets(signal=emg_data.to_numpy(), sampling_rate=1200)

    return o, t, emg_data


# We want to merge clusters who are in proximity to each other
def remove_outliers_by_x_axis_distance(clusters):
    clusters_to_remove = []
    t_clusters = []
    distances = []

    # Calculate median of distances
    for i in range(0, len(clusters) - 1):
        distances.append(abs(clusters[i].end - clusters[i + 1].start))
    prox_coef = np.median(distances) / 2

    for i in range(0, len(clusters) - 2):
        # Check for all clusters if the subsequent cluster is closer in proximity than median/2
        if abs(clusters[i].end - clusters[i + 1].start) < prox_coef:
            clusters[i].merge(clusters[i + 1])
            clusters_to_remove.append(clusters[i + 1])
            i += 1

    # Handle 'edge' case for last element of array
    if abs(clusters[-1].start - clusters[-2].end) < prox_coef:
        clusters[-2].merge(clusters[-1])
        clusters_to_remove.append(clusters[-1])

    if clusters_to_remove:
        get_logger().debug(f"Merged {len(clusters_to_remove)} clusters because of proximity")
    for cluster in clusters:
        if cluster not in clusters_to_remove:
            t_clusters.append(cluster)

    return t_clusters


def multi_dataset_onset_detection(datasets, config):
    for dataset in datasets:
        dataset.filtered_data = onset_detection(dataset, config)
