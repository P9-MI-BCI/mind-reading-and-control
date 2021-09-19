import random
from data_preprocessing.data_distribution import data_distribution


def train_test_split_data(data, split_per=10):
    dd = data_distribution(data)['expected_triggered_percent']
    isAcceptableDistribution = True

    while isAcceptableDistribution:
        random.shuffle(data)
        test_size = int(len(data) / 100 * 10)

        train_data = data[test_size:]
        test_data = data[:test_size]

        train_dd = data_distribution(train_data)['expected_triggered_percent']
        test_dd = data_distribution(test_data)['expected_triggered_percent']

        if train_dd < dd * 1.25 and train_dd > dd * 0.75:
            if test_dd < dd * 1.25 and test_dd > dd * 0.75:
                print(
                    f'The Train and Test Data has an acceptable triggered distribution of {train_dd} and {test_dd} percent - Returning.')
                isAcceptableDistribution = False
            else:
                print(
                    f'Test Data did not have an acceptable triggered distribution of {test_dd} percent - expected {dd * 0.75}-{dd * 1.25} percent, trying again.')
        else:
            print(
                f'Training Data did not have an acceptable triggered distribution of {train_dd} percent - expected {dd * 0.75}-{dd * 1.25} percent, trying again.')

    return train_data, test_data


def format_dataset(data, channel=0):
    # takes in list of dataframes on the usual format [target_value, pd.DataFrame] and returns these are seperate vectors for downstream prediction
    y = []
    x = []

    for frame in data:
        y.append(frame[0])
        x.append(frame[1][channel])

    # data, target
    return x, y 