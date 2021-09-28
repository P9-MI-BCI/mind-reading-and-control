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


