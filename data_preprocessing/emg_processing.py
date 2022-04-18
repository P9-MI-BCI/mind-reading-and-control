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


def onset_detection(dataset: Dataset, config, is_online=False, prox_coef=2, static_clusters=True,
                    proximity_outliers=True, iter_threshold=False, normalization=True) -> [[int]]:
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
        is_online=is_online, prox_coef=prox_coef, threshold=threshold, static_clusters=static_clusters,
        proximity_outliers=proximity_outliers, iter_threshold=iter_threshold, normalization=normalization)

    # Plotting of EMG signal and clusters
    t = [threshold] * len(filtered_data[config.EMG_CHANNEL])

    if get_logger().level == 10:
        cluster_plot_arr = []
        for cluster in emg_clusters:
            cluster_plot_arr.append(filtered_data[config.EMG_CHANNEL].iloc[cluster.start:cluster.end])

        fig = plt.figure(figsize=(40, 8))
        plt.plot(np.abs(filtered_data[config.EMG_CHANNEL]), color='black')

        for vals in cluster_plot_arr:
            plt.plot(np.abs(vals))

        plt.plot(t, '--', color='black')
        plt.title(dataset.filename)
        plt.xlabel('Time (s)')
        plt.ylabel('mV (Filtered)', labelpad=-2)
        # plt.autoscale()
        plt.show()

    dataset.clusters = emg_clusters
    dataset.filtered_data[config.EMG_CHANNEL] = filtered_data[config.EMG_CHANNEL]
    return filtered_data[config.EMG_CHANNEL]


def emg_clustering(emg_data, onsets: [int], distance=None, is_online=False, prox_coef=2, normalization=True,
                   threshold=None, iter=0, iter_threshold=True, static_clusters=True, proximity_outliers=True) -> [
    [int]]:
    if distance is None:
        distance = 100

    prev_cluster = []
    stop_loop = 0

    while True:
        clusters = []
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
                clusters.append(temp_clusters)
                temp_clusterobj = Cluster(data=temp_clusters)
                temp_clusterobj.create_info()
                cluster_list.append(temp_clusterobj)
                temp_clusters = [idx]

        # For last cluster
        clusters.append(temp_clusters)
        temp_clusterobj = Cluster(data=temp_clusters)
        temp_clusterobj.create_info()
        cluster_list.append(temp_clusterobj)

        # Check if no changes in previous iteration and break, else increase distance
        if (static_clusters):
            if len(clusters) == 20:
                break
            elif len(clusters) > 20:
                distance += 20
            elif len(clusters) < 20 and stop_loop < 20:
                stop_loop += 1
                distance -= 1
            else:
                break
        else:
            if prev_cluster == clusters:
                break
            else:
                prev_cluster = clusters
                distance += 100

    try:
        # Remove outliers by looking at their proximity
        if (proximity_outliers):
            assert len(clusters) > 2
            cluster_list = remove_outliers_by_x_axis_distance(cluster_list, prox_coef)

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
        if (normalization):
            onsets, threshold, emg_data = normalize_peaks(cluster_list, np.abs(emg_data))
            threshold = threshold
            emg_rectified = np.abs(emg_data) > threshold
            emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
            cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data), onsets=emg_onsets,
                                                               is_online=is_online, prox_coef=prox_coef,
                                                               normalization=False, threshold=threshold, iter=iter,
                                                               static_clusters=static_clusters,
                                                               proximity_outliers=proximity_outliers, iter_threshold=iter_threshold)

        if (iter_threshold):
            # If there are less than 20 clusters, increase threshold by 20% maximum 10 times
            if (len(cluster_list) < 20 and iter < 10):
                threshold += threshold * 0.20
                emg_rectified = np.abs(emg_data) > threshold
                emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
                cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data), onsets=emg_onsets,
                                                                   is_online=is_online, prox_coef=prox_coef,
                                                                   normalization=False, threshold=threshold,
                                                                   iter=iter + 1,
                                                                   static_clusters=static_clusters,
                                                                   proximity_outliers=proximity_outliers, iter_threshold=iter_threshold)
            # If there are 20 clusters, reduce threshold by 10% maximum 10 times
            elif (len(cluster_list) == 20 and iter < 10):
                threshold -= threshold * 0.05
                emg_rectified = np.abs(emg_data) > threshold
                emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
                cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data), onsets=emg_onsets,
                                                                   is_online=is_online, prox_coef=prox_coef,
                                                                   normalization=False, threshold=threshold,
                                                                   iter=iter + 1,
                                                                   static_clusters=static_clusters,
                                                                   proximity_outliers=proximity_outliers, iter_threshold=iter_threshold)

            # If we broke a dataset and it now does not have 20 clusters, increase threshold by 15% once
            elif ((len(cluster_list) != 20 and 1 < iter <= 10) or len(cluster_list) == 20 and iter == 10):
                print('ja')
                threshold += threshold * 0.15
                emg_rectified = np.abs(emg_data) > threshold
                emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
                cluster_list, emg_data, threshold = emg_clustering(emg_data=np.abs(emg_data), onsets=emg_onsets,
                                                                   is_online=is_online, prox_coef=prox_coef,
                                                                   normalization=False, threshold=threshold, iter=11,
                                                                   static_clusters=static_clusters,
                                                                   proximity_outliers=proximity_outliers, iter_threshold=iter_threshold)


        if is_online:
            return cluster_list, emg_data, threshold
        else:
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

    # Half along the y-axis any cluster with indices over Q3
    for cluster in clusters:
        if Q3 < emg_data[cluster.peak]:
            for index in cluster.data:
                while Q3 < emg_data[index]:
                    emg_data[index] = emg_data[index] / 2

        # Increase by the IQR Value along the y-axis for the clusters below Q1
        if Q1 > emg_data[cluster.peak]:
            for index in cluster.data:
                if Q1 > emg_data[index]:
                    emg_data[index] = emg_data[index] + iqr_val

    o, t = biosppy.signals.emg.find_onsets(signal=emg_data.to_numpy(), sampling_rate=1200)

    return o, t, emg_data


# Compare all peaks and remove outliers below Q1
# We don't care about outliers above Q3 as they have shown clear excess in force
def remove_outliers_by_peak_activity(clusters, emg_data):
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


# We want to merge clusters who are in close proximity to eachother
def remove_outliers_by_x_axis_distance(clusters, prox_coef):
    # TODO: Try median instead of prox_coef
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
                i += 1
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
        for dataset in datasets:
            temp = onset_detection(dataset, config, is_online)
            dataset.filtered_data[config.EMG_CHANNEL] = temp
            #dataset.filtered_data[config.EMG_CHANNEL] = temp
            #dataset.data = dataset.data.iloc[1:-1, :]
            #dataset.data.reset_index(drop=True, inplace=True)
