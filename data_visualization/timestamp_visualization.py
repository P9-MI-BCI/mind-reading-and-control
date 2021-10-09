import matplotlib.pyplot as plt
from classes import Frame
from definitions import OUTPUT_PATH


def visualize_frame(frame: Frame, config, freq: int, channel: int, num: int, save_fig: bool = False):
    x_seconds = []
    center = (len(frame.data) / 2) / freq
    for i, row in frame.filtered_data[channel].items():  # converts the frame.data freqs to seconds
        x_seconds.append(i / freq - center)

    if frame.label == 1:
        if config['aggregate_strategy'] == 'emg_start':
            emg_timestamp = [0, (frame.timestamp['emg_peak'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),
                         (frame.timestamp['emg_end'] - frame.timestamp[config['aggregate_strategy']]).total_seconds()]
            tp_timestamp = [(frame.timestamp['tp_start'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),
                            (frame.timestamp['tp_end'] - frame.timestamp[config['aggregate_strategy']]).total_seconds()]

        elif config['aggregate_strategy'] == 'emg_peak':
            emg_timestamp = [(frame.timestamp['emg_start'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),0,
                             (frame.timestamp['emg_end'] - frame.timestamp[config['aggregate_strategy']]).total_seconds()]
            tp_timestamp = [(frame.timestamp['tp_start'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),
                            (frame.timestamp['tp_end'] - frame.timestamp[config['aggregate_strategy']]).total_seconds()]
        elif config['aggregate_strategy'] == 'emg_end':
            emg_timestamp = [(frame.timestamp['emg_start'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),
                             (frame.timestamp['emg_peak'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(), 0]
            tp_timestamp = [(frame.timestamp['tp_start'] - frame.timestamp[config['aggregate_strategy']]).total_seconds(),
                            (frame.timestamp['tp_end'] - frame.timestamp[config['aggregate_strategy']]).total_seconds()]
        y_t = ['TP'] * len(tp_timestamp)
        y_t2 = ['EMG'] * len(emg_timestamp)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]}, sharex=True)

        ax1.set_title(f' Channel: {channel} - EEG {num+1} - Filter: {frame.filter_type[channel].iloc[0]}')
        ax1.plot(x_seconds, frame.filtered_data[channel], color='tomato')
        ax1.axvline(x=0, color='black', ls='--')

        ax2.set_title('EMG Detection')
        ax2.plot(emg_timestamp, y_t2, marker='^', color='limegreen')
        ax2.annotate('Peak', xy=[emg_timestamp[1], y_t2[1]])

        ax3.set_title('Trigger Point Duration')
        ax3.plot(tp_timestamp, y_t, marker='o', color='royalblue')
        ax3.annotate('Trigger Point', xy=[tp_timestamp[0], y_t[0]])

    else:
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]}, sharex=True)

        plt.title(f' Channel: {channel} - EEG Frame: {num + 1} - Filter: {frame.filter_type[channel].iloc[0]}')
        plt.plot(x_seconds, frame.filtered_data[channel], color='tomato')
        plt.axvline(x=0, color='black', ls='--')

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'{OUTPUT_PATH}/bandpass/all_average_emg_start/frames/{num + 1}.png')

    plt.show()
