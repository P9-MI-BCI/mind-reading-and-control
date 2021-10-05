import pandas as pd
from scipy.signal import butter, sosfilt, lfilter, filtfilt
from utility.logger import get_logger


def butter_sos_highpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff[0] / nyq
    return butter(N=order, Wn=normal_cutoff, analog=False, btype='highpass', output='sos')


def butter_highpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff[0] / nyq
    return butter(order, normal_cutoff, analog=False, btype='highpass', output='ba')


def butter_sos_bandpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_low = cutoff[0] / nyq
    normal_high = cutoff[1] / nyq
    return butter(N=order, Wn=[normal_low, normal_high], analog=False, btype='bandpass', output='sos')


def butter_bandpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_low = cutoff[0] / nyq
    normal_high = cutoff[1] / nyq
    return butter(N=order, Wn=[normal_low, normal_high], analog=False, btype='bandpass', output='ba')


def butter_sos_lowpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff[0] / nyq
    return butter(N=order, Wn=normal_cutoff, analog=False, btype='lowpass', output='sos')


def butter_lowpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff[0] / nyq
    return butter(N=order, Wn=normal_cutoff, analog=False, btype='lowpass', output='ba')


def butter_filter(data: pd.DataFrame, order: int = 5, cutoff=80, btype: str = 'highpass', freq: int = 1200):
    b, a, sos = None, None, None

    if (not isinstance(cutoff, list) and not isinstance(cutoff, tuple)):
        temp = []
        temp.append(cutoff)
        cutoff = temp

    if btype == 'sos_highpass' and len(cutoff) == 1:
        sos = butter_sos_highpass(order, cutoff, freq)
        return sosfilt(sos, data)

    elif btype == 'highpass' and len(cutoff) == 1:
        b, a = butter_highpass(order, cutoff, freq)
        return filtfilt(b, a, data)

    elif btype == 'sos_bandpass' and len(cutoff) == 2:
        sos = butter_sos_bandpass(order, cutoff, freq)
        return sosfilt(sos, data)

    elif btype == 'bandpass' and len(cutoff) == 2:
        b, a = butter_bandpass(order, cutoff, freq)
        return filtfilt(b, a, data)

    elif btype == 'sos_lowpass' and len(cutoff) == 1:
        sos = butter_sos_lowpass(order, cutoff, freq)
        return sosfilt(sos, data)

    elif btype == 'lowpass' and len(cutoff) == 1:
        b, a = butter_lowpass(order, cutoff, freq)
        return filtfilt(b, a, data)
    else:

        get_logger().error('Error in filter type or len(cutoff), check params')


