import numpy as np
from matplotlib import pyplot as plt

from main import init
from scipy.signal import butter, sosfilt, sosfreqz


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, analog=False, btype='high', output='sos')
    return sos


def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y


def plot_filtered_emg(order=5, freq=1200, cutoff=80):

    # Get the filter coefficients so we can check its frequency response.
    #sos = butter_highpass(cutoff, freq, order)

    # Plot the frequency response.
    # w, h = sosfreqz(sos, worN=2000)
    # plt.subplot(2, 1, 1)
    # plt.plot(0.5 * freq * w / np.pi, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5 * freq)
    # plt.title("High Filter Frequency Response")
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()

    dataset = init()
    data_pd = dataset.data_device1
    emg_data = data_pd[12]
    T = len(emg_data) / freq  # seconds
    n = int(T * freq)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_highpass_filter(emg_data, cutoff, freq, order)

    plt.subplot(2, 1, 1)
    plt.title("Original Data")
    plt.plot(t, emg_data, 'b-')
    # plt.xlim(31.3, 31.5)
    plt.subplot(2, 1, 2)
    plt.title("Filtered Data, Cutoff: "+str(cutoff))
    plt.plot(t, y, 'g-')
    # plt.xlim(31.3,31.5)
    plt.xlabel('Time [sec]')
    plt.grid()
    # plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()


plot_filtered_emg(cutoff=120)

