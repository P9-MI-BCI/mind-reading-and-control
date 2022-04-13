from statistics import mean, median
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.fft import fft
# take input windows and extract features for the XGBoost
from tqdm import tqdm
from scipy.stats import linregress


def extract_features(X):
    handcrafted_X = []

    for x in X:
        feature_vector = []
        for channel in x.T:
            feature_vector.append(mean(channel))
            feature_vector.append(skew(channel))
            feature_vector.append(kurtosis(channel))
            feature_vector.append(median(channel))
            feature_vector.append(peak_to_peak_time_window(channel))
            feature_vector.append(peak_to_peak_slope(channel))

        handcrafted_X.append(feature_vector)

    return np.array(handcrafted_X)


def _peak_to_peak(x):
    return max(x) - min(x)


def ts_max(x):
    return np.argmax(x)


def ts_min(x):
    return np.argmin(x)


def peak_to_peak_time_window(x):
    return ts_max(x) - ts_min(x)


def peak_to_peak_slope(x):
    return _peak_to_peak(x) / peak_to_peak_time_window(x)


