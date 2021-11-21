import biosppy
import numpy as np
import pandas as pd
from data_preprocessing.trigger_points import is_triggered
from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter
from utility.logger import get_logger
from statistics import mean, stdev, median
import matplotlib.pyplot as plt

# Clusters onsets based on time between each onset index.
# It starts with a very small time distance and increases the timespan between clusters until peaks_to_find is reached
def emg_clustering(emg_data: pd.DataFrame, onsets: [int], freq: int, referencing: bool = True,
                   cluster_range: float = 0.05, peaks_to_find: int = 30, tp_table: pd.DataFrame = pd.DataFrame()) \
        -> [[int]]:
    onset_clusters_array = []
    all_peaks = []

    if not referencing:
        while len(onset_clusters_array) != peaks_to_find:
            temp = []
            onset_clusters_array = []
            window = cluster_range * freq

            for i in onsets:
                if len(temp) == 0:
                    temp.append(i)
                elif abs(i - temp[-1]) < window:
                    temp.append(i)
                else:
                    onset_clusters_array.append(temp)
                    temp = []

            get_logger().debug(
                f'Found {len(onset_clusters_array)} clusters, if this is more than {peaks_to_find} then increment.')
            cluster_range += 0.01
            if len(onset_clusters_array) == 0:
                get_logger().error('CLUSTERS COULD NOT BE CREATED PROBABLY CHANGE PARAMETERS.')
                exit()

    # Discard all onsets that do not lie in a TriggerPoint interval
    elif referencing:
        window = cluster_range * freq
        temp = []
        referenced_onsets = []

        for i in onsets:
            if is_triggered(i, tp_table=tp_table):
                referenced_onsets.append(i)

        for i in referenced_onsets:
            if len(temp) == 0:
                temp.append(i)
            elif abs(i - temp[-1]) < window:
                temp.append(i)
            else:
                onset_clusters_array.append(temp)
                temp = []

        if len(onset_clusters_array) == 1:
            get_logger().error('CLUSTERS COULD NOT BE CREATED PROBABLY CHANGE PARAMETERS.')
            exit()

    else:
        get_logger().error('Did not enter if-statement, check \'referencing\' variable')
        exit()

    for onset_cluster in onset_clusters_array:
        highest = 0
        index = 0
        for onset in range(onset_cluster[0], onset_cluster[-1] + 1):
            if abs(emg_data[onset]) > highest:
                highest = abs(emg_data[onset])
                index = onset

        # saving start, peak, and end.
        all_peaks.append([onset_cluster[0], index, onset_cluster[-1]])

    '''
    Heuristic for removing TriggerPoint table entries that have no corresponding onset cluster.
    Also removes duplicate onset clusters for a single TriggerPoint interval (first come, first serve order)
    If referencing is True, will return a TP table and all_peaks array of equal length, where an element
    in either list corresponds to the same index element in the other.
    '''
    if referencing:
        save_arr = []
        dupe_arr = []
        tp_indexes = list(range(0, len(tp_table)))
        for i in range(0, len(all_peaks)):
            for j in tp_indexes:
                if (tp_table['tp_start'].iloc[j].total_seconds() * freq) < all_peaks[i][0] < (
                        tp_table['tp_end'].iloc[j].total_seconds() * freq):
                    if j not in save_arr:
                        save_arr.append(j)
                    else:
                        dupe_arr.append(i)

        not_save_arr = list(set(tp_indexes) - set(save_arr))
        for i in not_save_arr:
            tp_table.drop(i, inplace=True)

        tp_table.reset_index(inplace=True, drop=True)
        for i in range(len(all_peaks) - 1, -1, -1):
            if i in dupe_arr:
                all_peaks.__delitem__(i)

    return all_peaks


# Finds EMG onsets using highpass filtering and afterwards Biosppy's onset detection
def onset_detection(dataset: Dataset, tp_table: pd.DataFrame, config, bipolar_mode: bool) -> [[int]]:
    EMG_CHANNEL = 12
    # Filter EMG Data with specified butterworth filter params from config
    filtered_data = pd.DataFrame()
    if bipolar_mode:
        bipolar_emg = abs(dataset.data_device1[EMG_CHANNEL] - dataset.data_device1[EMG_CHANNEL + 1])
        filtered_data[EMG_CHANNEL] = butter_filter(data=bipolar_emg,
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )
    else:
        filtered_data[EMG_CHANNEL] = butter_filter(data=dataset.data_device1[EMG_CHANNEL],
                                                   order=config.emg_order,
                                                   cutoff=config.emg_cutoff,
                                                   btype=config.emg_btype,
                                                   )


    # Find onsets based on the filtered data
    onsets, threshold = biosppy.signals.emg.find_onsets(signal=filtered_data[EMG_CHANNEL].to_numpy(),
                                                  sampling_rate=dataset.sample_rate,
                                                  )

    emg_rectified = np.abs(filtered_data[config.EMG_Channel]) > threshold

    emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()

    # Group onsets based on time
    emg_clusters = emg_clustering_calibration(emg_data=filtered_data[EMG_CHANNEL],
                                              onsets=emg_onsets,
                                              peaks_to_find=len(tp_table),
                                             )

    return emg_clusters, filtered_data


def onset_threshold_detection(dataset: Dataset, tp_table: pd.DataFrame, config):
    EMG_CHANNEL = 12
    filtered_data = pd.DataFrame()

    # First highpass filtering
    filtered_data[EMG_CHANNEL] = butter_filter(data=dataset.data_device1[EMG_CHANNEL],
                                               order=config['emg_order'],
                                               cutoff=config['emg_cutoff'],
                                               btype=config['emg_btype'],
                                               )
    # Second rectify the signal
    emg_rectified = filtered_data.abs()

    # Third lowpass filtering
    filtered_data[EMG_CHANNEL] = butter_filter(data=emg_rectified[EMG_CHANNEL],
                                               order=config['emg2_order'],
                                               cutoff=config['emg2_cutoff'],
                                               btype=config['emg2_btype'],
                                               )

    emg_onsets = filtered_data[EMG_CHANNEL][filtered_data[EMG_CHANNEL] > config['emg_threshold']].index.tolist()

    # Group onsets based on time
    emg_clusters = emg_clustering(emg_data=filtered_data[EMG_CHANNEL],
                                  onsets=emg_onsets,
                                  freq=dataset.sample_rate,
                                  peaks_to_find=len(tp_table),
                                  tp_table=tp_table,
                                  referencing=False,
                                  cluster_range=0.05)

    return emg_clusters, filtered_data


def onset_detection_calibration(dataset: Dataset, config, input_peaks, bipolar_mode: bool = False) -> [[int]]:
    # Filter EMG Data with specified butterworth filter params from config
    filtered_data = pd.DataFrame()
    if bipolar_mode:
        bipolar_emg = abs(dataset.data_device1[config.EMG_Channel] - dataset.data_device1[config.EMG_Channel + 1])
        filtered_data[config.EMG_Channel] = butter_filter(data=bipolar_emg,
                                                          order=config.emg_order,
                                                          cutoff=config.emg_cutoff,
                                                          btype=config.emg_btype,
                                                          )
    else:
        filtered_data[config.EMG_Channel] = butter_filter(data=dataset.data_device1[config.EMG_Channel],
                                                          order=config.emg_order,
                                                          cutoff=config.emg_cutoff,
                                                          btype=config.emg_btype,
                                                          )
    # emg_rectified = np.abs(filtered_data[config.EMG_Channel])
    #
    # filtered_data[config.EMG_Channel] = butter_filter(emg_rectified,
    #                                                   order=3,
    #                                                   cutoff=5,
    #                                                   btype='lowpass')

    # plt.plot(filtered_data[config.EMG_Channel])
    # Find onsets based on the filtered data
    onsets, threshold = biosppy.signals.emg.find_onsets(signal=filtered_data[config.EMG_Channel].to_numpy(),
                                                       sampling_rate=dataset.sample_rate,
                                                       )
    # print(threshold)
    #
    t = [threshold] * len(filtered_data[config.EMG_Channel])
    emg_rectified = np.abs(filtered_data[config.EMG_Channel]) > threshold
    # emg_onsets = filtered_data[filtered_data[config.EMG_Channel] > threshold].index.tolist()
    # plt.plot(emg_rectified)
    plt.plot(filtered_data[config.EMG_Channel])
    plt.plot(t)

    plt.show()
    # print(onsets[0])
    emg_onsets = emg_rectified[emg_rectified == True].index.values.tolist()
    # Group onsets based on time
    emg_clusters = emg_clustering_calibration(emg_data=filtered_data[config.EMG_Channel],
                                              onsets=emg_onsets,
                                              peaks_to_find=input_peaks,
                                              )

    plot_arr = []
    for cluster in emg_clusters:
        plot_arr.append(filtered_data[config.EMG_Channel].iloc[cluster[0]:cluster[2]])

    plt.plot(filtered_data[config.EMG_Channel], color='black')
    for vals in plot_arr:
        plt.plot(vals)

    plt.plot(t)
    plt.show()

    return emg_clusters, filtered_data


def emg_clustering_calibration(emg_data: pd.DataFrame, onsets: [int], peaks_to_find: int = 30) -> [[int]]:
    all_peaks = []
    clusters = []
    temp_clusters = []
    for idx in onsets:
        if len(temp_clusters) == 0:
            temp_clusters.append(idx)
        elif abs(temp_clusters[-1] - idx) < 150:
            temp_clusters.append(idx)
        else:
            clusters.append(temp_clusters)
            temp_clusters = [idx]
    clusters.append(temp_clusters)
    lengths = [len(x) for x in clusters]
    lengths.sort()

    mean_cluster_size = mean(lengths)
    onset_clusters_array = [x for x in clusters if len(x) > mean_cluster_size]

    for onset_cluster in onset_clusters_array:
        highest = 0
        index = 0
        for onset in range(onset_cluster[0], onset_cluster[-1] + 1):
            if abs(emg_data[onset]) > highest:
                highest = abs(emg_data[onset])
                index = onset

        # saving start, peak, and end.
        # fifth_percent = int(len(onset_cluster)*0.05)
        all_peaks.append([onset_cluster[0], index, onset_cluster[-1]])

    return all_peaks
