import pandas as pd

from data_training.SVM.svm_prediction import svm_classifier
from utility.logger import get_logger
from scipy.stats import linregress


def svm_training_hub(x_train, y_train, x_test, y_test):
    feature_vec = create_feature_vector(x_train)

    result = svm_classifier(x_train, y_train, x_test, y_test, features=feature_vec)


"""
Divides each window in X into seven sub-windows and extracts linregress slope, variability, and mean amplitude 
features from them.
"""
def create_feature_vector(X):
    window_sz = len(X[0])  # window size in seconds
    sub_window_sz = int(0.5 * 1200)  # sliding windows are 500 ms
    step_sz = int(sub_window_sz / 2)  # step size is 50% overlap
    feature_vector = pd.DataFrame()

    for x in X:
        window = pd.DataFrame(x)
        for channel in window.columns:
            feature_vec = []
            amount_sub_windows = window_sz / sub_window_sz * 2 - 1
            for sw in range(0, window_sz - step_sz, step_sz):
                freq_range = sw, sw + sub_window_sz
                data = window.iloc[freq_range[0]:freq_range[1], :]

                x = data[channel].index.values
                y = data[channel].values

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                var = data[channel].var()
                mean_amp = data[channel].mean()
                feature_vec.append([slope, var, mean_amp])

            feature_vector[channel] = [feature_vec]
            assert len(feature_vec) == amount_sub_windows

    return feature_vector
