import numpy as np
import pandas as pd

from classes import Dataset
import neurokit2 as nk



def blink_detection(data: pd.DataFrame, sample_rate: int, EOG_CHANNEL: int = 9) -> np.array:
    eog_signal = nk.as_vector(data[EOG_CHANNEL])
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=sample_rate, method='neurokit')

    blinks = nk.eog_findpeaks(eog_cleaned, sampling_rate=sample_rate, method="mne")

    return blinks
