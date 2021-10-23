import biosppy.signals
import biosppy.plotting
from classes import Dataset
import pandas as pd
from utility.logger import get_logger


# Clusters onsets based on time between each onset index.
# It starts with a very small time distance and increases the timespan between clusters until peaks_to_find is reached
def emg_clustering(emg_data: pd.DataFrame, onsets: [int], freq: int, peaks_to_find: int) -> [[int]]:
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

        # saving start, peak, and end.
        all_peaks.append([onset_cluster[0], index, onset_cluster[-1]])

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
                                                   order=config['emg_order'],
                                                   cutoff=config['emg_cutoff'],
                                                   btype=config['emg_btype'],
                                                   )

    # Find onsets based on the filtered data
    onsets, = biosppy.signals.emg.find_onsets(signal=filtered_data[EMG_CHANNEL].to_numpy(),
                                              sampling_rate=dataset.sample_rate,
                                              )

    # Group onsets based on time
    emg_clusters = emg_clustering(emg_data=filtered_data[EMG_CHANNEL],
                                  onsets=onsets,
                                  freq=dataset.sample_rate,
                                  peaks_to_find=len(tp_table),
                                  )

    return emg_clusters, filtered_data
