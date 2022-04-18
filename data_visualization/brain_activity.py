import matplotlib.pyplot as plt

from classes.Dataset import Dataset


def visualize_brain_activity(datasets: [Dataset], config):
    channels = config.EEG_CHANNELS
    window_sz = int((2 * 1200) / 2)

    for dataset in datasets:
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(dataset.filename)

        for i, onset in enumerate(dataset.onsets_index):
            ax = plt.subplot(6, 4, i + 1)

            center = onset[0]
            data = dataset.filtered_data.iloc[center - window_sz: center + window_sz].iloc[:, :9].T
            im = plt.pcolor(data)

            ax.pcolor(data)
            ax.set_yticks(range(len(channels)))
            ax.set_yticklabels(channels)
            ax.set_title(f'Onset {i}')
            # fig.subplots_adjust(wspace=0, hspace=0)
            ax.axvline(1200, ls='--', color='black')

            # Onsets
            # onsets = plt.subplot(gs[1], sharex=heatmap)

            # Color bar
            cbar = fig.colorbar(ax=ax, mappable=im, )
            cbar.set_label('Amplitude, uV')

        fig.tight_layout()
        plt.show()
