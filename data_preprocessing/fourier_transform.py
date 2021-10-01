import pandas as pd
import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift
from utility.logger import get_logger


def fourier_transform_single_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:

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
def fourier_transform_listof_dataframes(lst_dataframe: [pd.DataFrame]) -> [pd.DataFrame]:

    lst_transformed = []

    for frame in lst_dataframe:
        columns = len(frame.data.columns)

        transformed = []
        for channel in frame.data:
            # get_logger().info("Fourier transforming channel: " + str(channel))
            fourier = fft(frame.data[channel].values)
            transformed.append(np.abs(fourier))

        transformed = pd.DataFrame(transformed)
        transformed = transformed.transpose()
        frame.data = transformed
        lst_transformed.append(frame)
        if len(frame.data.columns) != columns:
            get_logger().warning("Failed to fourier transform all channels. Transformed " +
                                 str(len(frame.data.columns)) + ", expected: " + str(columns))
        # get_logger().info("Fourier transformed " + str(len(frame.data.columns)) + " channels, returning")

    return lst_transformed



