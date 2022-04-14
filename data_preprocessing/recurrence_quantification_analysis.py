import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

from scipy.spatial.distance import pdist, squareform
from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid
from definitions import OUTPUT_PATH
from utility.file_util import create_dir

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator


def recurrence_quantification(training_data, config, plot=True):
    left_hemisphere = ['T7', 'C5', 'C3']
    right_hemisphere = ['C4', 'C6', 'T8']
    longitudinal_fissure = ['C1', 'Cz', 'C2']

    window_sz = int((2 * config.SAMPLE_RATE) / 2)

    for dataset in training_data:
        lhs = dataset.filtered_data[left_hemisphere]
        rhs = dataset.filtered_data[right_hemisphere]
        lf = dataset.filtered_data[longitudinal_fissure]

        for i, onset in enumerate(dataset.onsets_index):
            center = onset[0]
            if center - window_sz < 0:
                continue

            lhs_win = lhs.iloc[center - window_sz: onset[2] + window_sz].T
            rhs_win = rhs.iloc[center - window_sz: onset[2] + window_sz].T
            lf_win = lf.iloc[center - window_sz: onset[2] + window_sz].T

            lhs_res = calc_rqa(lhs_win)
            rhs_res = calc_rqa(rhs_win)
            lf_res = calc_rqa(lf_win)





                # if plot:
                #     fig = plt.figure(figsize=(10, 5))
                #     im = ImageGenerator.generate_recurrence_plot(info['Recurrence_Matrix'])
                #     plt.plot(im)
                #     plt.show()



                # time_series = TimeSeries(X[0],
                #                          embedding_dimension=2,
                #                          time_delay=3)
                # settings = Settings(time_series,
                #                     analysis_type=Classic,
                #                     neighbourhood=FixedRadius(0.000002),
                #                     similarity_measure=EuclideanMetric,
                #                     theiler_corrector=1)
                # computation = RQAComputation.create(settings,
                #                                     verbose=False)
                #
                # result = computation.run()
                # result.min_diagonal_line_length = 2
                # result.min_vertical_line_length = 2
                # result.min_white_vertical_line_length = 2
                # print(result)
                #
                # if plot:
                #     rp = RPComputation.create(settings, verbose=False).run()
                #    # plt.imshow(rp.recurrence_matrix_reverse_normalized)
                #     path = os.path.join(OUTPUT_PATH, 'recurrence_plots')
                #     filename = f'recurrence_plot_{dataset.filename}_onset{i}.png'
                #     #create_dir(path, recursive=True)
                #     im = ImageGenerator.generate_recurrence_plot(rp.recurrence_matrix_reverse)
                #     ImageGenerator.save_recurrence_plot(rp.recurrence_matrix_reverse, os.path.join(path, filename))

                # rp = RecurrencePlot(threshold='point', percentage=20)
                #
                # lhs_rp = rp.transform(X)
                #
                # plt.title(f'{dataset.filename} - onset_{i}')
                # plt.imshow(lhs_rp[0], cmap='binary', origin='lower')
                # plt.show()


def calc_rqa(data_window):
    results = []
    for i, X in data_window.iterrows():
        X_results, X_info = nk.complexity_rqa(X, show=False)
        results.append({f'X{i}_res': X_results, f'X{i}_inf': X_info})

    return results


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
    d = pdist(s[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


def moving_average(s, r=5):
    return np.convolve(s, np.ones((r,)) / r, mode='valid')
