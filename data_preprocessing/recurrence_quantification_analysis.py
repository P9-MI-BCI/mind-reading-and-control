import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid



def recurrence_quantification(training_data, config):
    left_hemisphere = ['T7', 'C5', 'C3']
    right_hemisphere = ['C4', 'C6', 'T8']
    longitudinal_fissure = ['C1', 'Cz', 'C2']

    window_sz = int((2 * 1200) / 2)

    for dataset in training_data:
        lhs = dataset.filtered_data[left_hemisphere]
        rhs = dataset.filtered_data[right_hemisphere]
        lf = dataset.filtered_data[longitudinal_fissure]

        for i, onset in enumerate(dataset.onsets_index):
            center = onset[0]
            lhs_win = lhs.iloc[center - window_sz: center + window_sz].T
            rhs_win = rhs.iloc[center - window_sz: center + window_sz].T
            lf_win = lf.iloc[center - window_sz: center + window_sz].T

            fig = plt.figure(figsize=(10, 5))

            for index, row in lhs_win.iterrows():
                X = np.array([row])

                rp = RecurrencePlot(threshold='point', percentage=20)

                lhs_rp = rp.transform(X)

                plt.title(f'{dataset.filename} - onset_{i}')
                plt.imshow(lhs_rp[0], cmap='binary', origin='lower')
                plt.show()



def binary_recurrence_matrix(channel_subset: pd.DataFrame()):
    recurrence_threshold = 0.3

    recurrence_matrix = []
    for index, row in channel_subset.iterrows():
        temp_matrix = []
        i, j = 0, 1
        while j < len(row):
            temp = recurrence_threshold - abs(row[i] - row[j])
            i += 1
            j += 1
            if temp < 0:
                temp_matrix.append(1)
            else:
                temp_matrix.append(0)
        recurrence_matrix.append(temp_matrix)

    return recurrence_matrix


def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z


def moving_average(s, r=5):
    return np.convolve(s, np.ones((r,))/r, mode='valid')


