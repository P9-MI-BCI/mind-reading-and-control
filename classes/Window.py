import pandas as pd
from utility.logger import get_logger
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import gridspec
from definitions import OUTPUT_PATH
from utility.save_figure import save_figure
import os


class Window:
    negative_slope = pd.DataFrame()
    variability = pd.DataFrame()
    mean_amplitude = pd.DataFrame()
    signal_negativity = pd.DataFrame()

    def __int__(self, label: int = 0, blink: int = 0, data: pd.DataFrame = 0, timestamp: pd.Series = 0, frequency_range=0,
                filtered_data: pd.DataFrame = 0, filter_type: pd.DataFrame = 0, num_id=0, aggregate_strategy=0):
        self.label = label
        self.blink = blink
        self.data = data
        self.timestamp = timestamp
        self.frequency_range = frequency_range
        self.filtered_data = filtered_data
        self.num_id = num_id
        self.aggregate_strategy = aggregate_strategy
        self.negative_slope = pd.DataFrame()
        self.variability = pd.DataFrame()
        self.mean_amplitude = pd.DataFrame()
        self.signal_negativity = pd.DataFrame()
        self.filter_type = pd.DataFrame()

    def filter(self, filter_in, channel: int, **kwargs):
        try:
            try:
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.filtered_data[channel], **kwargs))
            except:
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
                self.filter_type = pd.DataFrame()
                self.filter_type[channel] = [filter_types[channel].iloc[0]]

    def _calc_negative_slope(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.negative_slope = pd.DataFrame()
            for channel in self.filtered_data.columns:
                x = self.filtered_data[channel].idxmin(), self.filtered_data[channel].idxmax()
                y = self.filtered_data[channel].min(), self.filtered_data[channel].max()

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                self.negative_slope[channel] = [slope]
        else:
            get_logger().error("Cannot feature extract negative slope without filtered data.")

    def _calc_variability(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.variability = pd.DataFrame()
            for channel in self.filtered_data.columns:
                self.variability[channel] = [self.filtered_data[channel].var()]
        else:
            get_logger().error("Cannot feature extract variability without filtered data.")

    def _calc_mean_amplitude(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.mean_amplitude = pd.DataFrame()
            for channel in self.filtered_data.columns:
                self.mean_amplitude[channel] = [self.filtered_data[channel].mean()]
        else:
            get_logger().error("Cannot feature extract mean amplitude without filtered data.")

    def _calc_signal_negativity(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.signal_negativity = pd.DataFrame()
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

                self.signal_negativity[channel] = [sum_negativity]
        else:
            get_logger().error("Cannot feature extract signal negativity without filtered data.")

    def extract_features(self):
        self._calc_negative_slope()
        self._calc_variability()
        self._calc_mean_amplitude()
        self._calc_signal_negativity()

    def get_features(self):
        existing_features = []
        if len(self.negative_slope) > 0:
            existing_features.append("negative_slope")
        if len(self.mean_amplitude) > 0:
            existing_features.append("mean_amplitude")
        if len(self.variability) > 0:
            existing_features.append("variability")
        if len(self.signal_negativity) > 0:
            existing_features.append("signal_negativity")

        return existing_features

    def plot(self, channel=4, freq=1200, show=True, plot_features=False, save_fig=False, overwrite=False) -> plt.figure():
        x_seconds = []
        fig = plt.figure(figsize=(5, 7))
        center = (len(self.data) / 2) / freq
        for i, row in self.filtered_data[channel].items():  # converts the window.data freqs to seconds
            x_seconds.append(i / freq - center)

        if self.label == 1:
            agg_strat = self.aggregate_strategy

            emg_timestamp, tp_timestamp = self._timestamp_order(agg_strat)
            y_t = ['EC'] * len(tp_timestamp)
            y_t2 = ['EMG'] * len(emg_timestamp)

            gs = gridspec.GridSpec(ncols=1, nrows=6, figure=fig)
            ax1 = fig.add_subplot(gs[:2, 0])
            ax1.set_title(f' Channel: {channel + 1} - EEG {self.num_id + 1} - Raw')
            ax1.plot(x_seconds, self.data[channel], color='tomato')
            ax1.axvline(x=0, color='black', ls='--')

            ax4 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
            ax4.set_title(f'Filter: {self.filter_type[channel].iloc[0]}')
            ax4.plot(x_seconds, self.filtered_data[channel], color='tomato', label='filtered data')
            if plot_features:
                x, y, slope = self._plot_features(center, freq, channel)
                ax4.plot(x, y, label=f'slope {round(slope,9)}', alpha=0.7)
                mean = self.filtered_data[channel].mean()
                y_mean = [mean] * len(self.filtered_data)
                ax4.plot(x_seconds, y_mean, label=f'mean {round(self.filtered_data[channel].mean(), 7)}', alpha=0.7)

                x_arr, y_arr, sum_negative = self._plot_signal_negativity(center, freq, channel)

                ax4.plot(x_arr[0], y_arr[0], color='black', label=f'negative {round(sum_negative, 7)}', alpha=0.7)
                for xi, yi in zip(x_arr, y_arr):
                    ax4.plot(xi, yi, color='black', alpha=0.7)

                ax4.legend()
            ax4.axvline(x=0, color='black', ls='--')

            ax2 = fig.add_subplot(gs[4, 0], sharex=ax1)
            ax2.set_title('EMG Detection')
            ax2.plot(emg_timestamp, y_t2, marker='^', color='limegreen')
            ax2.annotate('Peak', xy=[emg_timestamp[1], y_t2[1]])

            ax3 = fig.add_subplot(gs[5, 0], sharex=ax1)
            ax3.set_title('Execution Cue Interval')
            ax3.plot(tp_timestamp, y_t, marker='o', color='royalblue')
            ax3.annotate('Execution Cue', xy=[tp_timestamp[0], y_t[0]])

        else:
            gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title(
                f' Channel: {channel + 1} - EEG Window: {self.num_id + 1} - Filter: Raw')
            ax1.plot(x_seconds, self.data[channel], color='tomato')
            ax1.axvline(x=0, color='black', ls='--')

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_title(f'Filter: {self.filter_type[channel].iloc[0]}')
            ax2.plot(x_seconds, self.filtered_data[channel], color='tomato')
            if plot_features:
                x, y, slope = self._plot_features(center, freq, channel)
                ax2.plot(x, y, label='slope', alpha=0.7)

                y_mean = [self.filtered_data[channel].mean()] * len(self.filtered_data)
                ax2.plot(x_seconds, y_mean, label='mean', alpha=0.7)

                x_arr, y_arr, sum_negative = self._plot_signal_negativity(center, freq, channel)

                ax2.plot(x_arr[0], y_arr[0], color='black', label=f'negative', alpha=0.7)
                for xi, yi in zip(x_arr, y_arr):
                    ax2.plot(xi, yi, color='black', alpha=0.7)
                ax2.legend()
            ax2.axvline(x=0, color='black', ls='--')

        plt.tight_layout()

        if save_fig:
            path = f'{OUTPUT_PATH}/plots/window_plots/channel{channel}_{self.num_id + 1}.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=overwrite)
            except FileExistsError:
                get_logger().exception(f'Found file already exists: {file} you can '
                                       f'overwrite the file by setting overwrite=True')

        if show:
            plt.show()

        return fig

    def _timestamp_order(self, agg_strat):
        emg_timestamp = []
        tp_timestamp = []

        if agg_strat == 'emg_start':
            emg_timestamp = [0, (
                    self.timestamp['emg_peak'] - self.timestamp[agg_strat]).total_seconds(),
                             (self.timestamp['emg_end'] - self.timestamp[
                                 agg_strat]).total_seconds()]
            tp_timestamp = [
                (self.timestamp['tp_start'] - self.timestamp[agg_strat]).total_seconds(),
                (self.timestamp['tp_end'] - self.timestamp[agg_strat]).total_seconds()]

        elif agg_strat == 'emg_peak':
            emg_timestamp = [
                (self.timestamp['emg_start'] - self.timestamp[agg_strat]).total_seconds(), 0,
                (self.timestamp['emg_end'] - self.timestamp[agg_strat]).total_seconds()]
            tp_timestamp = [
                (self.timestamp['tp_start'] - self.timestamp[agg_strat]).total_seconds(),
                (self.timestamp['tp_end'] - self.timestamp[agg_strat]).total_seconds()]
        elif agg_strat == 'emg_end':
            emg_timestamp = [
                (self.timestamp['emg_start'] - self.timestamp[agg_strat]).total_seconds(),
                (self.timestamp['emg_peak'] - self.timestamp[agg_strat]).total_seconds(), 0]
            tp_timestamp = [
                (self.timestamp['tp_start'] - self.timestamp[agg_strat]).total_seconds(),
                (self.timestamp['tp_end'] - self.timestamp[agg_strat]).total_seconds()]

        return emg_timestamp, tp_timestamp

    def _plot_features(self, center, freq, channel):
        x1, x2 = self.filtered_data[channel].idxmin(), self.filtered_data[channel].idxmax()
        y = self.filtered_data[channel].min(), self.filtered_data[channel].max()

        slope, intercept, r_value, p_value, std_err = linregress([x1, x2], y)
        return [x1 / freq - center, x2 / freq - center], y, slope

    def _plot_signal_negativity(self, center, freq, channel):
            prev = self.filtered_data[channel].iloc[0]
            sum_negativity = 0
            res_x, res_y = [], []
            consecutive_negative_x = []
            consecutive_negative_y = []

            for i, value in self.filtered_data[channel].items():
                diff = prev - value

                if diff > 0:
                    sum_negativity -= diff
                    prev = value
                    consecutive_negative_x.append(i / freq - center)
                    consecutive_negative_y.append(value)

                else:
                    res_x.append(consecutive_negative_x)
                    res_y.append(consecutive_negative_y)
                    prev = value

                    consecutive_negative_x = []
                    consecutive_negative_y = []

            return res_x, res_y, sum_negativity
