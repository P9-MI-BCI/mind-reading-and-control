import pandas as pd
from scipy.signal import butter, sosfilt
from utility.logger import get_logger


def butter_helper(order, cutoff, btype, freq):
    nyq = 0.5 * freq
    sos = None
    if btype == 'highpass' and len(cutoff) == 1:
        normal_cutoff = cutoff[0] / nyq
        sos = butter(order, normal_cutoff, analog=False, btype=btype, output='sos')
    elif btype == 'bandpass' and len(cutoff) == 2:
        normal_low = cutoff[0] / nyq
        normal_high = cutoff[1] / nyq
        sos = butter(N=order, Wn=(normal_low, normal_high), btype=btype, fs=freq, output='sos')
    elif btype == 'lowpass' and len(cutoff) == 1:
        normal_cutoff = cutoff[0] / nyq
        sos = butter(order, normal_cutoff, analog=False, btype=btype, output='sos')
    else:
        get_logger().error('Unsupported filter type, select \'highpass\' with len(cutoff) == 1, or \'bandpass\' with len(cutoff) == 2')
    return sos


def butter_filter(data: pd.DataFrame, order: int = 4, cutoff=None, btype: str = 'highpass', freq: int = 1200):
    if cutoff is None:
        cutoff = [100]
    sos = butter_helper(order, cutoff, btype, freq)
    y = sosfilt(sos, data)
    return y


