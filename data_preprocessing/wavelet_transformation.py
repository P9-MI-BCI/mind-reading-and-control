import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

from classes.Dataset import Dataset


def multi_dataset_wavelet_transformation(training_data, config, wavelet_type: str, plot=False):
    wavelet_features = []
    # for dataset in training_data:
    #     wavelet_features.append(wavelet_transformation(dataset, config, wavelet_type, plot))
    wavelet_transformation(training_data, config, wavelet_type, plot=plot)

    return wavelet_features


def wavelet_transformation(training_data: [Dataset], config, wavelet_type: str, plot=False):
    sampling_rate = config.SAMPLE_RATE
    window_sz = 25 #int((2 * sampling_rate) / 2)

    wavelet_coeffs = []
    for dataset in training_data:
        onset_window_coeffs = []
        for onset in dataset.onsets_index:
            center = onset[0]
            if center - window_sz < 0:
                continue

            data = dataset.data[config.EEG_CHANNELS].iloc[center - window_sz: center + window_sz]

            totalscal = 64  # scale
            wavename = 'morl'
            fc = pywt.central_frequency(wavename)  # central frequency
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(1, totalscal + 1)
            sampling_period = 1.0 / sampling_rate
            time = len(dataset.data) / sampling_rate
            t = np.arange(0, time, 1.0 / sampling_rate)

            # Discrete wavelet transform. Can either be single level or multilevel.
            if wavelet_type == 'discrete':
                cA, cD = discrete_wavelet_transform(data, 'db4', level=0)
                coeffs = discrete_wavelet_transform(data, 'db4', level=5)

                if plot:
                    max_lev = 3  # how many levels of decomposition to draw
                    label_levels = 3  # how many levels to explicitly label on the plots
                    shape = data.shape

                    fig, axes = plt.subplots(2, 4, figsize=(25, 25))
                    for level in range(0, max_lev + 1):
                        if level == 0:
                            # show the original image before decomposition
                            axes[0, 0].set_axis_off()
                            axes[1, 0].imshow(data, cmap=plt.cm.gray)
                            axes[1, 0].set_title('Image')
                            axes[1, 0].set_axis_off()
                            continue

                        # plot subband boundaries of a standard DWT basis
                        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                                         label_levels=label_levels)
                        axes[0, level].set_title('{} level\ndecomposition'.format(level))

                        # compute the 2D DWT
                        c = pywt.wavedec2(data, 'db4', mode='periodization', level=level)
                        # normalize each coefficient array independently for better visibility
                        c[0] /= np.abs(c[0]).max()
                        for detail_level in range(level):
                            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
                        # show the normalized coefficients
                        arr, slices = pywt.coeffs_to_array(c)
                        axes[1, level].imshow(arr, cmap=plt.cm.gray)
                        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
                        axes[1, level].set_axis_off()

                    plt.tight_layout()
                    plt.show()

                    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
                    # ax1.plot(dataset)
                    # ax2.plot(cA, '-g')
                    # ax3.plot(cD, '-r')
                    # fig.suptitle(f'{dataset.filename}')
                    #
                    # plt.show()

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


def parsevals_theorem(data):
    energy_sum = 0
    for i in range(len(data)):
        energy_sum += abs(data[i])**2

    return energy_sum


def discrete_wavelet_transform(channel_data, mode: str, level: int):
    # PyWavelet computation of the maximum useful level of decomposition
    # max_level = pywt.dwt_max_level(len(channel_data), 'db4')

    if level > 0:
        coeffs = pywt.wavedec2(channel_data, mode, level=level)
        return coeffs  # Returns: [cA_n, cD_n, cD_n-1, …, cD2, cD1] : list
    else:
        cA, cD = pywt.dwt(channel_data, mode)
        return cA, cD


def continuous_wavelet_transform(channel_data, scales, wavename, sampling_period):
    cwtmatr, freqs = pywt.cwt(channel_data, scales, wavelet=wavename, sampling_period=sampling_period)

    return cwtmatr, freqs
