import numpy as np
import pandas as pd
import neurokit2 as nk


# Firstly cleans the data (filtering) to easilier find the blink artifacts.
# Then they find the peaks, which denotes blinks. Returns array of frequencies for blinks.
def blink_detection(data: pd.DataFrame, sample_rate: int) -> np.array:
    EOG_CHANNEL = 9

    eog_signal = nk.as_vector(data[EOG_CHANNEL])
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=sample_rate, method='neurokit')
    blinks = nk.eog_findpeaks(eog_cleaned, sampling_rate=sample_rate, method="mne")

    return blinks
