import random

from utility.logger import get_logger
import pandas as pd
import numpy as np


def format_dataset(data: [pd.DataFrame], channel=0, features='raw') -> ([], []):
    # takes in list of datawindows on the usual format [target_value, pd.DataFrame] and returns these as separate
    # vectors for downstream prediction
    y = []
    x = []

    if len(data) < 1:
        get_logger().warning('List of windows was empty while attempting to format data and target variables - Exiting')
        exit()
    if features == 'raw':
        for window in data:
            y.append(window.label)
            x.append(window.data.iloc[channel])
    elif features == 'filtered':
        for window in data:
            y.append(window.label)
            x.append(window.filtered_data.iloc[channel])
    elif features == 'features':
        for window in data:
            feats = window.get_features()
            y.append(window.label)

            temp_features = []
            for feat in feats:
                df = getattr(window, feat)
                temp_features.append(df[channel].item())
            x.append(temp_features)
    elif features == 'feature_vec':
        for window in data:
            y.append(window.label)
            x.append(np.array(window.feature_vector[channel].iloc[0]).flatten())


    # data, target
    return x, y
