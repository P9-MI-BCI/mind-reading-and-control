import pandas as pd


def calc_best_fit_slope(filtered_data: pd.DataFrame, channel):

    X = filtered_data[channel].index.values
    Y = filtered_data[channel].values

    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)

    n = len(X)  # or len(Y)

    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi ** 2 for xi in X]) - n * xbar ** 2

    b = numer / denum
    a = ybar - b * xbar

    return a, b


def calc_variability(filtered_data: pd.DataFrame, channel):
    return filtered_data[channel].var()


def calc_mean_amplitude(filtered_data: pd.DataFrame, channel):
    return filtered_data[channel].mean()

