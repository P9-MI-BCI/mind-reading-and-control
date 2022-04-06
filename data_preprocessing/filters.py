import pandas as pd
from scipy.signal import butter, filtfilt, lfilter, iirnotch

from classes.Dataset import Dataset
from utility.logger import get_logger


def butter_highpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, analog=False, btype='highpass', output='ba')


def butter_bandpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_low = cutoff[0] / nyq
    normal_high = cutoff[1] / nyq
    return butter(N=order, Wn=[normal_low, normal_high], analog=False, btype='bandpass', output='ba')


def butter_lowpass(order, cutoff, freq):
    nyq = 0.5 * freq
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, analog=False, btype='lowpass', output='ba')


def notch(freq, remove_freq, quality_factor=30):
    return iirnotch(remove_freq, quality_factor, freq)


def butter_filter(data: pd.DataFrame, order: int, cutoff, btype: str, freq: int = 1200):
    if btype == 'highpass':
        b, a = butter_highpass(order, cutoff, freq)
        return filtfilt(b, a, data)

    elif btype == 'bandpass' and len(cutoff) == 2:
        b, a = butter_bandpass(order, cutoff, freq)
        return filtfilt(b, a, data)

    elif btype == 'lowpass':
        b, a = butter_lowpass(order, cutoff, freq)
        return filtfilt(b, a, data)

    elif btype == 'notch':
        b, a = notch(freq, cutoff)
        return filtfilt(b, a, data)
    else:
        get_logger().error('Error in filter type or len(cutoff), check params', btype)


def data_filtering(filter_range, config, dataset: Dataset):
    try:
        assert dataset
        filtered_data = pd.DataFrame()
        for channel in config.EEG_CHANNELS:
            filtered_data[channel] = butter_filter(data=dataset.data[channel],
                                                   order=config.EEG_ORDER,
                                                   cutoff=filter_range,
                                                   btype=config.EEG_BTYPE,
                                                   freq=dataset.sample_rate)

        return filtered_data[config.EEG_CHANNELS]
    except AssertionError:
        get_logger().exception(f'{data_filtering.__name__} received an empty dataset.')


def multi_dataset_filtering(filter_range, config, datasets):
    if len(datasets) > 1:
        for dataset in datasets:
            dataset.filtered_data[config.EEG_CHANNELS] = data_filtering(filter_range, config, dataset)
    else:
        datasets.filtered_data[config.EEG_CHANNELS] = data_filtering(filter_range, config, datasets)
