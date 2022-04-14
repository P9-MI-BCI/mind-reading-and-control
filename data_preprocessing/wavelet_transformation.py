import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

from classes.Dataset import Dataset


def multi_dataset_wavelet_transformation(training_data, config, wavelet_type: str, plot=False):
    wavelet_features = []
    for dataset in training_data:
        wavelet_features.append(wavelet_transformation(dataset, config, wavelet_type, plot))

    return wavelet_features


def wavelet_transformation(dataset: Dataset, config, wavelet_type: str, plot=False):
    sampling_rate = config.SAMPLE_RATE
    window_sz = int((2 * sampling_rate) / 2)

    wavelet_coeffs = []
    for channel in dataset.filtered_data:
        onset_window_coeffs = []
        for onset in dataset.onsets_index:
            center = onset[0]
            if center - window_sz < 0:
                continue

            data = dataset.filtered_data[channel].iloc[center - window_sz: onset[2] + window_sz]

            totalscal = 64  # scale
            wavename = 'morl'
            fc = pywt.central_frequency(wavename)  # central frequency
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(1, totalscal + 1)
            sampling_period = 1.0 / sampling_rate
            time = len(dataset.filtered_data[channel]) / sampling_rate
            t = np.arange(0, time, 1.0 / sampling_rate)

            # Discrete wavelet transform. Can either be single level or multilevel.
            if wavelet_type == 'discrete':
                cA, cD = discrete_wavelet_transform(data, 'db4', level=0)
                coeffs = discrete_wavelet_transform(data, 'db4', level=5)

                if plot:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
                    ax1.plot(dataset)
                    ax2.plot(cA, '-g')
                    ax3.plot(cD, '-r')
                    fig.suptitle(f'{dataset.filename} - {channel}')

                    plt.show()

            # Continuous wavelet transform
            if wavelet_type == 'continuous':
                cwtmatr, freqs = continuous_wavelet_transform(data, scales, wavename, sampling_period)

                wavelet_coeffs.append(cwtmatr)

                if plot:
                    fig = plt.figure(1)
                    plt.contourf(t, freqs, abs(cwtmatr))
                    plt.ylabel(u"freq(Hz)")
                    plt.xlabel(u"time(s)")
                    plt.colorbar()
                    # plt.ylim(0, 60)
                    plt.show()

    return wavelet_coeffs


def discrete_wavelet_transform(channel_data, mode: str, level: int):
    # PyWavelet computation of the maximum useful level of decomposition
    # max_level = pywt.dwt_max_level(len(channel_data), 'db4')

    if level > 0:
        coeffs = pywt.wavedec(channel_data, mode, level=level)
        return coeffs  # Returns: [cA_n, cD_n, cD_n-1, …, cD2, cD1] : list
    else:
        cA, cD = pywt.dwt(channel_data, mode)
        return cA, cD


def continuous_wavelet_transform(channel_data, scales, wavename, sampling_period):
    cwtmatr, freqs = pywt.cwt(channel_data, scales, wavelet=wavename, sampling_period=sampling_period)

    return cwtmatr, freqs
