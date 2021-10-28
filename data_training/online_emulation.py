from tqdm import tqdm
from classes.Dataset import Dataset
from classes.Window import Window
from data_preprocessing.filters import butter_filter
import pandas as pd
from data_training.measurements import get_accuracy, get_precision, get_recall, get_f1_score
from utility.logger import get_logger


# Simulates online behavior by consuming the dataset sequentially (windows)
def simulate_online(dataset: Dataset, config, models, features: str = 'features', continuous=False, data_buffer=True):
    WINDOW_LENGTH = 2  # seconds
    WINDOW_SIZE = WINDOW_LENGTH * dataset.sample_rate
    EEG_CHANNELS = list(range(0, 9))
    EMG_CHANNEL = 12

    continuous_data = pd.DataFrame(columns=EEG_CHANNELS)
    data_buffer_size = WINDOW_SIZE * 10

    all_window_preds = []
    all_windows = []
    if data_buffer:
        get_logger().info('Building Data Buffer.')

    for i in tqdm(range(0, len(dataset.data_device1), WINDOW_SIZE)):
        predictions = []
        window = Window()
        window.data = dataset.data_device1.iloc[i:i + WINDOW_SIZE]
        window.label = None
        window.num_id = i
        window.frequency_range = [i, i + WINDOW_SIZE]

        # will use previous knowledge of seen data
        if continuous:
            if data_buffer:
                # keeps the buffer at a minimum size and will not run following code until satisfied.
                if len(continuous_data) < data_buffer_size:
                    continuous_data = pd.concat(
                        [continuous_data, dataset.data_device1.iloc[i:i + WINDOW_SIZE, EEG_CHANNELS]],
                        ignore_index=True)
                    continue
                else:
                    continuous_data = pd.concat(
                        [continuous_data, dataset.data_device1.iloc[i:i + WINDOW_SIZE, EEG_CHANNELS]])
                filtered_data = pd.DataFrame(columns=EEG_CHANNELS)

                # filters the new window + the previous continuous data
                for chan in EEG_CHANNELS:
                    filtered_data[chan] = butter_filter(data=continuous_data[chan],
                                                        order=config['eeg_order'],
                                                        cutoff=config['eeg_cutoff'],
                                                        btype=config['eeg_btype']
                                                        )

                window.filtered_data = filtered_data.iloc[-WINDOW_SIZE:]
                window.filtered_data = window.filtered_data.reset_index(drop=True)
                # removes the first window to keep buffer size small
                continuous_data = continuous_data.iloc[WINDOW_SIZE:]

        # other case where you only process one window at a time and doesnt consider previous data
        else:
            for channel in EEG_CHANNELS:
                window.filter(butter_filter,
                              channel,
                              order=config['eeg_order'],
                              cutoff=config['eeg_cutoff'],
                              btype=config['eeg_btype'],
                              freq=dataset.sample_rate
                              )

        filter_type_df = pd.DataFrame(columns=[EMG_CHANNEL], data=[config['emg_btype']])
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

            # predicts using a saved model for each channel, with the selected feature
            predictions.append(models[channel].predict([feature_vector]).tolist()[0])

        all_windows.append(window)
        all_window_preds.append(predictions)

    # returns the windows created during runtime and all predictions made
    return all_windows, all_window_preds


# evaluates the online performance on the selected channels
def evaluate_online_predictions(windows: [Window], predictions: [int], index: [(int, int)], channels=None):
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

    # iterates through each window with the corresponding predictions.
    # checks if the windows' frequency is within the pair_index (start, end) list
    # (which determines if the window is labeled as mrcp)
    for window, prediction in zip(windows, predictions):
        frequency_range = [window.frequency_range[0], window.frequency_range[1]]

        pred = 0
        # majority vote for prediction
        for chan in channels:
            pred += prediction[chan]
        if pred > len(channels) // 2:
            predictions_z.append(1)
        else:
            predictions_z.append(0)

        is_in_index = False

        # traverses the index list by checking all statements
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

    return score_dict
