import biosppy.signals
import biosppy.plotting
import numpy as np
from main import init
import math

def find_emg_peaks(channel=12, window=2.2, type=None):

    if type is None:
        return
    dataset = init()
    data_pd = dataset.data_device1
    emg_data = data_pd[channel]
    freq = dataset.sample_rate

    ts, filtered, onsets = biosppy.signals.emg.emg(emg_data, sampling_rate=freq, show=False)

    temp = []
    onset_clusters_array = []
    all_peaks = []
    window = window*freq

    for i in onsets:
        if len(temp) == 0:
            temp.append(i)
        elif abs(i - temp[-1]) < window:
            temp.append(i)
        else:
            onset_clusters_array.append(temp)
            temp = []

    if type == 'highest':
        for onset_cluster in onset_clusters_array:
            highest = 0
            index = 0
            for onset in onset_cluster:
                if abs(emg_data[onset]) > highest:
                    highest = abs(emg_data[onset])
                    index = onset
            all_peaks.append(index)

    elif type == 'first':
        for onset_cluster in onset_clusters_array:
            all_peaks.append(onset_cluster[0])

    #biosppy.plotting.plot_emg(ts=ts,sampling_rate=1200,raw=emg_data,filtered=filtered,onsets=all_peaks,show=True)
    return all_peaks




#peaks = find_emg_peaks(window=2.15,type='first')

