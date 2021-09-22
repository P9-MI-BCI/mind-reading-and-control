import pandas as pd
import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift
from utility.logger import get_logger


def fourier_transform_single_dataframe(dataframe):

    transformed = []
    columns = len(dataframe.columns)

    for channel in dataframe:
        get_logger().info("Fourier transforming channel: " + str(channel))
        fourier = fft(dataframe[channel].values)
        transformed.append(np.abs(fourier))

    transformed = pd.DataFrame(transformed)
    transformed = transformed.transpose()

    if len(transformed.columns) != columns:
        get_logger().warning("Failed to fourier transform all channels. Transformed " +
                             str(len(transformed.columns)) + ", expected: " + str(columns))
    get_logger().info("Fourier transformed " + str(len(dataframe.columns)) + " channels, returning")

    return transformed


# Add logger, Christoffer help
def fourier_transform_listof_dataframes(dataframe):

    lst_transformed = []

    for frame in dataframe:
        transformed = []
        for channel in frame.data:
            get_logger().info("Fourier transforming channel: " + str(channel))
            fourier = fft(frame.data[channel].values)
            transformed.append(np.abs(fourier))

        transformed = pd.DataFrame(transformed)
        transformed = transformed.transpose()
        frame.data = transformed
        lst_transformed.append(frame)


    return lst_transformed

# if __name__ == '__main__':
#
#     dataset = init()
#     frequency = dataset.sample_rate
#     data_pd = dataset.data_device1
#
#     # print(fourier_transform(data_pd))
#
#     N = data_pd.shape[0]
#     T = 1.0 / frequency
#     y = data_pd[15].values
#
#     #x = np.linspace(0.0, N * T, N, endpoint=False)
#
#
#
#     yf = fft(y)
#     xf = fftfreq(N, T)
#
#     plt.plot(xf, np.abs(yf))
#
#     plt.show()

