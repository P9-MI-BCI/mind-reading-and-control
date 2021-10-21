# emulate an online environment where the dataset is processed sequentially, the last 20% of the dataset should be used a test-set?
# labelling will be difficult.
from classes.Dataset import Dataset
from classes.Window import Window
from data_preprocessing.data_distribution import slice_and_label_idle_windows
from data_preprocessing.filters import butter_filter
import numpy as np
import pandas as pd

from data_training.measurements import combine_predictions


def emulate_online(dataset: Dataset, config, models):
    WINDOW_LENGTH = 2  # seconds
    WINDOW_SIZE = WINDOW_LENGTH * dataset.sample_rate
    EEG_CHANNELS = list(range(0, 9))

    all_window_preds = []
    all_windows = []
    for i in range(0, len(dataset.data_device1), WINDOW_SIZE):
        window = Window()
        predictions = []
        window.data = dataset.data_device1.iloc[i:i + WINDOW_SIZE]
        window.label = None
        window.num_id = i
        window.frequency_range = [i, i+WINDOW_SIZE]

        for channel in EEG_CHANNELS:
            window.filter(butter_filter,
                          channel,
                          order=config['eeg_order'],
                          cutoff=config['eeg_cutoff'],
                          btype=config['eeg_btype'],
                          freq=dataset.sample_rate
                          )

        filter_type_df = pd.DataFrame(columns=[12], data=[config['emg_btype']])
        filter_type_df[EEG_CHANNELS] = [config['eeg_btype']] * len(EEG_CHANNELS)

        window.update_filter_type(filter_type_df)

        window.extract_features()

        for channel in EEG_CHANNELS:
            feature_vector = []

            for feature in window.get_features():
                f = getattr(window, feature)
                feature_vector.append(f[channel].item())

            predictions.append(models[channel].predict([feature_vector]).tolist()[0])

        all_windows.append(window)
        all_window_preds.append(predictions)

    return all_windows, all_window_preds


def evaluate_online_predictions(windows, predictions, index, channels=None):
    if channels is None:
        channels = list(range(0, 9))

    correct_predictions = []
    for window, prediction in zip(windows, predictions):
        frequency_range = list(range(window.frequency_range[0], window.frequency_range[1]))

        pred = 0
        for chan in channels:
            pred += prediction[chan]
        if pred > len(channels)//2:
            pred = 1
        else:
            pred = 0

        is_in_index = False
        freq_optimize = 0
        for i in frequency_range:
            if i in index:
                is_in_index = True
                freq_optimize = i

        if not is_in_index:
            freq_optimize = window.frequency_range[1]

        if is_in_index and pred:
            correct_predictions.append(1)
        elif is_in_index and not pred:
            correct_predictions.append(0)
        elif not is_in_index and pred:
            correct_predictions.append(0)
        else:
            correct_predictions.append(1)

        index = [x for x in index if x >= freq_optimize]

    return correct_predictions

