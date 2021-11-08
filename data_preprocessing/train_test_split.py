import random
from data_preprocessing.data_distribution import data_distribution

from utility.logger import get_logger
import pandas as pd


# Creates a test and train split with the same label distribution as the input data
def train_test_split_data(data: [pd.DataFrame], split_per: int = 10) -> ([pd.DataFrame], [pd.DataFrame]):
    dd = data_distribution(data)['expected_labeled_percent']
    isAcceptableDistribution = True
    upper_bound = 1.10
    lower_bound = 0.90

    counter = 0
    # continuously shuffles the data until the distribution of the train and test split is acceptable
    while isAcceptableDistribution and counter < 100:
        random.shuffle(data)
        test_size = int(len(data) / 100 * split_per)

        train_data = data[test_size:]
        test_data = data[:test_size]

        train_dd = data_distribution(train_data)['expected_labeled_percent']
        test_dd = data_distribution(test_data)['expected_labeled_percent']

        if dd * upper_bound > train_dd > dd * lower_bound:
            if dd * upper_bound > test_dd > dd * lower_bound:
                get_logger().info(
                    f'The Train and Test Data has an acceptable label distribution of {train_dd} and {test_dd} '
                    f'percent - Returning.')
                isAcceptableDistribution = False
            else:
                get_logger().debug(
                    f'Test Data did not have an acceptable label distribution of {test_dd} percent '
                    f'- expected {round(dd * lower_bound, 2)}-{round(dd * upper_bound, 2)} percent, trying again.')
        else:
            get_logger().debug(
                f'Training Data did not have an acceptable label distribution of {train_dd} percent '
                f'- expected {round(dd * lower_bound, 2)}-{round(dd * upper_bound, 2)} percent, trying again.')
        counter += 1

    if counter >= 99:
        get_logger().warning('Tried to find acceptable label distribution more than 100 times.')
        exit()

    if len(train_data) < 1 or len(test_data) < 1:
        get_logger().warning('Train or test split was created with size < 1.')

    return train_data, test_data


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

    # data, target
    return x, y
