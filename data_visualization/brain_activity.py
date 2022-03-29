import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from classes.Dataset import Dataset


def visualize_brain_activity(datasets: [Dataset], config):
    channels = config.EEG_CHANNELS

    data = datasets[0].data

    data = []
    for channel in datasets[0].filtered_data:
        data.append(datasets[0].filtered_data[channel])

    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    #ax.set_xticks(np.arange(len(data)))
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels)

    fig.tight_layout()
    plt.show()
