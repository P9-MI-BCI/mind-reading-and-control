import random
from data_preprocessing.data_distribution import data_distribution

from utility.logger import get_logger
import pandas as pd


def train_test_split_data(data: [pd.DataFrame], split_per: int=10) -> ([pd.DataFrame], [pd.DataFrame]):
    dd = data_distribution(data)['expected_triggered_percent']
    isAcceptableDistribution = True
    upper_bound = 1.10
    lower_bound = 0.90

    while isAcceptableDistribution:
        random.shuffle(data)
        test_size = int(len(data) / 100 * split_per)

        train_data = data[test_size:]
        test_data = data[:test_size]

        train_dd = data_distribution(train_data)['expected_triggered_percent']
        test_dd = data_distribution(test_data)['expected_triggered_percent']

        if dd * upper_bound > train_dd > dd * lower_bound:
            if dd * upper_bound > test_dd > dd * lower_bound:
                get_logger().info(
                    f'The Train and Test Data has an acceptable label distribution of {train_dd} and {test_dd} percent - Returning.')
                isAcceptableDistribution = False
            else:
                get_logger().debug(
                    f'Test Data did not have an acceptable label distribution of {test_dd} percent - expected {dd * lower_bound}-{dd * upper_bound} percent, trying again.')
        else:
            get_logger().debug(
                f'Training Data did not have an acceptable label distribution of {train_dd} percent - expected {dd * lower_bound}-{dd * upper_bound} percent, trying again.')

    if len(train_data) < 1 or len(test_data) < 1:
        get_logger().warning('Train or test split was created with size < 1.')

    return train_data, test_data


def format_dataset(data: [pd.DataFrame], channel=0) -> ([], []):
    # takes in list of datawindows on the usual format [target_value, pd.DataFrame] and returns these are seperate vectors for downstream prediction
    y = []
    x = []

    if len(data) < 1:
        get_logger().warning('List of Frames was empty while attempting to format data and target variables.')
    for window in data:
        y.append(window.label)
        x.append(window.data.iloc[channel])

    # data, target
    return x, y


