import biosppy.signals
import biosppy.plotting
from classes import Dataset
import pandas as pd
from utility.logger import get_logger


def emg_clustering(emg_data: pd.DataFrame, onsets, freq: int, peaks_to_find: int) -> []:
    onset_clusters_array = []
    all_peaks = []

    cluster_range = 0.05
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

        get_logger().debug(f'Found {len(onset_clusters_array)} clusters if this is more than {peaks_to_find} then increment.')
        cluster_range += 0.01
        if len(onset_clusters_array) == 1:
            get_logger().error('CLUSTERS COULD NOT BE CREATED PROBABLY CHANGE PARAMETERS.')
            break

    for onset_cluster in onset_clusters_array:
        highest = 0
        index = 0
        for onset in onset_cluster:
            if abs(emg_data[onset]) > highest:
                highest = abs(emg_data[onset])
                index = onset

        all_peaks.append([onset_cluster[0], index, onset_cluster[-1]])

    return all_peaks

