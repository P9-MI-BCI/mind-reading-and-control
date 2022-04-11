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

    wavelet_coeffs = []
    for channel in dataset.filtered_data:

        # Discrete wavelet transform
        if wavelet_type == 'discrete':
            # cA, cD = discrete_wavelet_transform(dataset.filtered_data[channel], level=0)
            coeffs = discrete_wavelet_transform(dataset.filtered_data[channel], level=5)

            energy_distributions = []
            for c in coeffs:
                energy_distributions.append(parsevals_theorem(c))


            if plot:
                # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
                # ax1.plot(dataset.filtered_data[channel])
                # ax2.plot(cA, '-g')
                # ax3.plot(cD, '-r')
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
                # for c in coeffs:
                #     plt.bar(c, len(c))
                # fig.suptitle(f'{dataset.filename} - {channel}')
                # plt.show()

        # Continuous wavelet transform
        if wavelet_type == 'continuous':
            totalscal = 64  # scale
            wavename = 'morl'
            fc = pywt.central_frequency(wavename)  # central frequency
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(1, totalscal + 1)
            sampling_period = 1.0 / sampling_rate
            time = len(dataset.filtered_data[channel]) / sampling_rate
            t = np.arange(0, time, 1.0 / sampling_rate)

            cwtmatr, freqs = continuous_wavelet_transform(dataset.filtered_data[channel], scales, wavename,
                                                          sampling_period)

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


def parsevals_theorem(data):
    energy_sum = 0
    for i in range(len(data)):
        energy_sum += abs(data[i])**2

    return energy_sum


def discrete_wavelet_transform(channel_data, level: int):
    max_level = pywt.dwt_max_level(len(channel_data), 'db4')


    if level > 0:
        # Returns: [cA_n, cD_n, cD_n-1, …, cD2, cD1] : list
        coeffs = pywt.wavedec(channel_data, 'db4', level=level)
        return coeffs
    else:
        cA, cD = pywt.dwt(channel_data, 'db4')
        return cA, cD


def continuous_wavelet_transform(channel_data, scales, wavename, sampling_period):
    cwtmatr, freqs = pywt.cwt(channel_data, scales, wavelet=wavename, sampling_period=sampling_period)

    return cwtmatr, freqs
