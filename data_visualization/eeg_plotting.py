import matplotlib.pyplot as plt
from definitions import OUTPUT_PATH


def plot_eeg(raw_data, filtered_data, label: str, all: bool = True):
    for i in range(0, 9):
        plt.plot(raw_data[i], label='raw data')
        plt.plot(filtered_data[i], label=label)
        plt.title(f'Channel: {i + 1}')
        plt.legend()
        if all:
            plt.savefig(f'{OUTPUT_PATH}/{label}/all/all_{label}_channel{i + 1}.png')
        else:
            plt.savefig(f'{OUTPUT_PATH}/{label}/frames/frame_{label}_channel{i + 1}.png')
        plt.show()