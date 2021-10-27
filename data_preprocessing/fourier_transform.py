import pandas as pd
import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift
from utility.logger import get_logger


# Takes in a single channel and transforms it.
def fourier_transform_single_data_channel(datawindow: pd.DataFrame) -> pd.DataFrame:

    transformed = []

    fourier = fft(datawindow.values)
    transformed.append(np.abs(fourier))

    transformed = pd.DataFrame(transformed)
    transformed = transformed.transpose()

    return transformed



# Add logger, Christoffer help
def fourier_transform_listof_datawindows(lst_data_window: [pd.DataFrame]) -> [pd.DataFrame]:

    lst_transformed = []

    for window in lst_data_window:
        columns = len(window.data.columns)

        transformed = []
        for channel in window.data:
            # get_logger().info("Fourier transforming channel: " + str(channel))
            fourier = fft(window.data[channel].values)
            transformed.append(np.abs(fourier))

        transformed = pd.DataFrame(transformed)
        transformed = transformed.transpose()
        window.data = transformed
        lst_transformed.append(window)
        if len(window.data.columns) != columns:
            get_logger().warning("Failed to fourier transform all channels. Transformed " +
                                 str(len(window.data.columns)) + ", expected: " + str(columns))
        # get_logger().info("Fourier transformed " + str(len(window.data.columns)) + " channels, returning")

    return lst_transformed



