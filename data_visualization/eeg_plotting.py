import matplotlib.pyplot as plt
from definitions import OUTPUT_PATH
from classes import Dataset


def plot_eeg(data: Dataset, label: str, savefig: bool = False):
    for i in range(0, 9):
        plt.plot(data.data_device1[i], label='raw data')
        plt.plot(data.filtered_data[i], label=label)
        plt.title(f'Channel: {i + 1}')
        plt.legend()

        if savefig:
            plt.savefig(f'{OUTPUT_PATH}/{label}/{label}_channel{i + 1}.png')

        plt.show()
