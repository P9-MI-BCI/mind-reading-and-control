import matplotlib.pyplot as plt
from classes import Dataset, Window


def visualize_windows(data: Dataset, windows: [Window], channel: int = 4, xlim: int = 400000):

    fig = plt.figure(figsize=(10, 6))

    for window in windows:
        start = window.frequency_range[0]
        end = window.frequency_range[-1]

        if window.label == 1 and window.blink == 0:
            mrcp_win = plt.axvspan(start, end, color='green', alpha=0.5, label='MRCP')
        elif window.label == 1 and window.blink == 1:
            mrcp_blink_win = plt.axvspan(start, end, color='red', alpha=0.5, label='MRCP w. blink')
        elif window.label == 0 and window.blink == 0:
            idle_win = plt.axvspan(start, end, color='grey', alpha=0.5, label='Idle')
        elif window.label == 0 and window.blink == 1:
            idle_blink_win = plt.axvspan(start, end, color='black', alpha=0.5, label='Idle w. blink')

    plt.xlabel('Frequency')
    plt.plot(data.filtered_data[channel])
    plt.xlim(0, xlim)
    plt.legend(handles=[mrcp_win, mrcp_blink_win, idle_win, idle_blink_win], loc='center', bbox_to_anchor=(1.15, 0.5))
    plt.title(f'Windows of channel: {channel}')
    fig.tight_layout()
    plt.show()
