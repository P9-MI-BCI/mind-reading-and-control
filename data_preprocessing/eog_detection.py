import numpy as np
from classes import Dataset
import neurokit2 as nk


def blink_detection(data: Dataset, EOG_CHANNEL: int = 9) -> np.array:
    eog_signal = nk.as_vector(data.data_device1[EOG_CHANNEL])
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=data.sample_rate, method='neurokit')

    blinks = nk.eog_findpeaks(eog_cleaned, sampling_rate=data.sample_rate, method="mne")

    return blinks
