import pandas as pd
from utility.logger import get_logger
from scipy.stats import linregress


class Window:

    negative_slope = pd.DataFrame()
    variability = pd.DataFrame()
    mean_amplitude = pd.DataFrame()
    signal_negativity = pd.DataFrame()
    filter_type = pd.DataFrame()

    def __int__(self, label: int = 0, data: pd.DataFrame = 0, timestamp: pd.Series = 0,
                filtered_data: pd.DataFrame = 0, filter_type: pd.DataFrame = 0):
        self.label = label
        self.data = data
        self.timestamp = timestamp
        self.filtered_data = filtered_data

    def filter(self, filter_in, channel: int, **kwargs):
        try:
            try:
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.filtered_data[channel], **kwargs))
            except KeyError:
                get_logger().debug('Adding key to filtered data window.')
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.data[channel], **kwargs))
        except AttributeError:
            get_logger().debug('filtered data was not yet initialized, creating data window.')
            self.filtered_data = pd.DataFrame(filter_in(self.data[channel], **kwargs))

    def update_filter_type(self, filter_types: pd.DataFrame):
        for channel in filter_types:
            try:
                try:
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
                except KeyError:
                    get_logger().debug('Key did not yet exist in filter type, adding it.')
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
            except AttributeError:
                get_logger().debug('Attribute not yet created in a window - initializing data window.')
                self.filter_type[channel] = [filter_types[channel].iloc[0]]

    def _calc_negative_slope(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            for channel in self.filtered_data.columns:
                x = self.filtered_data[channel].idxmin(), self.filtered_data[channel].idxmax()
                y = self.filtered_data[channel].min(), self.filtered_data[channel].max()

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                self.negative_slope[channel] = [slope]

    def _calc_variability(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            for channel in self.filtered_data.columns:
                self.variability[channel] = [self.filtered_data[channel].var()]

    def _calc_mean_amplitude(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            for channel in self.filtered_data.columns:
                self.mean_amplitude[channel] = [self.filtered_data[channel].mean()]

    def _calc_signal_negativity(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            for channel in self.filtered_data.columns:
                prev = self.filtered_data[channel].iloc[0]
                sum_negativity = 0

                for i, value in self.filtered_data[channel].items():
                    diff = prev - value
                    if diff < 0:
                        sum_negativity += diff
                        prev = value
                    else:
                        prev = value

                self.signal_negativity[channel] = sum_negativity

    def extract_features(self):
        self._calc_negative_slope()
        self._calc_variability()
        self._calc_mean_amplitude()
        self._calc_signal_negativity()

