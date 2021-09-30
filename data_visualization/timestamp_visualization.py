import matplotlib.pyplot as plt
from classes import Frame


def visualize_frame(frame: Frame, freq: int, channel: int):
    tp_timestamp = [frame.timestamp['tp_start'].total_seconds() * freq, frame.timestamp['tp_end'].total_seconds() * freq]
    emg_timestamp = [frame.timestamp['emg_start'].total_seconds() * freq, frame.timestamp['emg_peak'].total_seconds() * freq, frame.timestamp['emg_end'].total_seconds() * freq]
    y_t = ['TP'] * len(tp_timestamp)
    y_t2 = ['EMG'] * len(emg_timestamp)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.set_title(f'Channel: {channel} - Frame')
    ax1.plot(frame.data[channel], color='tomato')
    ax2.set_title('EMG Detection')
    ax2.plot(emg_timestamp, y_t2, marker='^', color='limegreen')
    ax2.annotate('Peak', xy=[emg_timestamp[1], y_t2[1]])
    ax3.set_title('Trigger Point Duration')
    ax3.plot(tp_timestamp, y_t, marker='o', color='royalblue')
    ax3.annotate('Trigger Point', xy=[tp_timestamp[0], y_t[0]])

    plt.subplots_adjust(hspace=0.35)
    plt.show()
