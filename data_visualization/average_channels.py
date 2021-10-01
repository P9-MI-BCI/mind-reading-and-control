from classes import Frame
import pandas as pd
import matplotlib.pyplot as plt


def average_channel(frames: [Frame]) -> Frame:
    test = [0,2,3,5,6,7,8,9,10,11,13,14,15,17,19,20,24,28]
    columns = frames[0].filtered_data.columns
    avg_channel = []
    for col in columns:
        df_temp = pd.DataFrame()
        # for i in range(0, len(frames)):
        #     df_temp[i] = frames[i].filtered_data[col]
        for i in test:
            df_temp[i] = frames[i].filtered_data[col]

        frame = Frame.Frame()
        frame.data = df_temp.mean(axis=1)
        avg_channel.append(frame)

    return avg_channel
    # takes in list of all frames with mrcp and returns a frame containing the average of them.


def plot_average_channels(avg_channels):

    for channel in range(0, len(avg_channels)):
        plt.plot(avg_channels[channel].data)
        plt.title(f'Channel: {channel}')
        plt.show()
