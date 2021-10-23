# emulate an online environment where the dataset is processed sequentially, the last 20% of the dataset should be used a test-set?
# labelling will be difficult.
from tqdm import tqdm
import matplotlib.pyplot as plt
from classes.Dataset import Dataset
from classes.Window import Window
from data_preprocessing.data_distribution import cut_and_label_idle_windows
from data_preprocessing.filters import butter_filter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_preprocessing.mrcp_detection import surrogate_channels
from data_training.measurements import combine_predictions, get_accuracy, get_precision, get_recall, get_f1_score
from utility.logger import get_logger


def emulate_online(dataset: Dataset, config, models, features='features', continuous=False, data_buffer=True):
    WINDOW_LENGTH = 2  # seconds
    WINDOW_SIZE = WINDOW_LENGTH * dataset.sample_rate
    EEG_CHANNELS = list(range(0, 9))

    continuous_data = pd.DataFrame(columns=EEG_CHANNELS)
    data_buffer_size = WINDOW_SIZE * 10

    all_window_preds = []
    all_windows = []
    if data_buffer:
        get_logger().info('Building Data Buffer.')

    for i in tqdm(range(0, len(dataset.data_device1), WINDOW_SIZE)):
        window = Window()
        predictions = []
        window.data = dataset.data_device1.iloc[i:i + WINDOW_SIZE]
        window.label = None
        window.num_id = i
        window.frequency_range = [i, i + WINDOW_SIZE]

        if continuous:
            if data_buffer:
                if len(continuous_data) < data_buffer_size:
                    continuous_data = pd.concat(
                        [continuous_data, dataset.data_device1.iloc[i:i + WINDOW_SIZE, EEG_CHANNELS]],
                        ignore_index=True)
                    continue
                else:
                    continuous_data = pd.concat(
                        [continuous_data, dataset.data_device1.iloc[i:i + WINDOW_SIZE, EEG_CHANNELS]])
                filtered_data = pd.DataFrame(columns=EEG_CHANNELS)

                for chan in EEG_CHANNELS:
                    filtered_data[chan] = butter_filter(data=continuous_data[chan],
                                                        order=config['eeg_order'],
                                                        cutoff=config['eeg_cutoff'],
                                                        btype=config['eeg_btype']
                                                        )

                window.filtered_data = filtered_data.iloc[-WINDOW_SIZE:]
                window.filtered_data = window.filtered_data.reset_index(drop=True)
                continuous_data = continuous_data.iloc[WINDOW_SIZE:]

        else:
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

            if features == 'raw':
                feature_vector = window.data.iloc[channel]
            elif features == 'filtered':
                feature_vector = window.filtered_data.iloc[channel]
            elif features == 'features':
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

    score_dict = pd.DataFrame()
    score_dict.index = ['accuracy',
                        'precision',
                        'recall',
                        'f1'
                        ]
    predictions_z = []
    target_z = []
    i_p = 0
    for window, prediction in zip(windows, predictions):
        frequency_range = [window.frequency_range[0], window.frequency_range[1]]

        pred = 0
        for chan in channels:
            pred += prediction[chan]
        if pred > len(channels) // 2:
            predictions_z.append(1)
        else:
            predictions_z.append(0)

        is_in_index = False

        traverse = True
        while traverse:
            if i_p >= len(index):
                is_in_index = False
                traverse = False
            elif frequency_range[0] > index[i_p][0]:
                if frequency_range[0] < index[i_p][1]:
                    is_in_index = True
                    traverse = False
                else:
                    i_p += 1
            elif frequency_range[0] < index[i_p][0]:
                is_in_index = False
                traverse = False
            elif frequency_range[0] < index[i_p][1]:
                i_p += 1

        if is_in_index:
            target_z.append(1)
        else:
            target_z.append(0)

    score_dict['online'] = [get_accuracy(target_z, predictions_z),
                            get_precision(target_z, predictions_z),
                            get_recall(target_z, predictions_z),
                            get_f1_score(target_z, predictions_z),
                            ]

    print(predictions_z)
    print(target_z)
    print(confusion_matrix(target_z, predictions_z))

    return score_dict
