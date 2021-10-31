import pandas as pd

from data_preprocessing.eog_detection import blink_detection
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

    def __int__(self, label: int = 0, blink: int = 0, data: pd.DataFrame = 0, timestamp: pd.Series = 0,
                frequency_range=None, is_sub_window=False, sub_windows=0,
                filtered_data: pd.DataFrame = 0, filter_type: pd.DataFrame = 0, num_id=0, aggregate_strategy=0):
        if frequency_range is None:
            frequency_range = []
        self.label = label
        self.blink = blink
        self.data = data
        self.timestamp = timestamp
        self.frequency_range = frequency_range
        self.is_sub_window = is_sub_window
        self.filtered_data = filtered_data
        self.num_id = num_id
        self.sub_windows = sub_windows
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
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.data[channel], **kwargs))
        except AttributeError:
            self.filtered_data = pd.DataFrame(filter_in(self.data[channel], **kwargs))

    def update_filter_type(self, filter_types: pd.DataFrame):
        for channel in filter_types:
            try:
                try:
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
                except KeyError:
                    get_logger().exception('Key did not yet exist in filter type, adding it.')
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
            except AttributeError:
                self.filter_type = pd.DataFrame()
                self.filter_type[channel] = [filter_types[channel].iloc[0]]

    def blink_detection(self, blinks):
        self.blink = 0
        for blink in blinks:
            if blink in range(self.frequency_range[0], self.frequency_range[1]):
                self.blink = 1  # indicates blink within window

    def _calc_negative_slope(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.negative_slope = pd.DataFrame()
            for channel in self.filtered_data.columns:
                x = self.filtered_data[channel].idxmin(), self.filtered_data[channel].idxmax()
                y = self.filtered_data[channel].min(), self.filtered_data[channel].max()

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                self.negative_slope[channel] = [slope]
        else:
            get_logger().error('Cannot feature extract negative slope without filtered data.')

    def _calc_variability(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.variability = pd.DataFrame()
            for channel in self.filtered_data.columns:
                self.variability[channel] = [self.filtered_data[channel].var()]
        else:
            get_logger().error('Cannot feature extract variability without filtered data.')

    def _calc_mean_amplitude(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.mean_amplitude = pd.DataFrame()
            for channel in self.filtered_data.columns:
                self.mean_amplitude[channel] = [self.filtered_data[channel].mean()]
        else:
            get_logger().error('Cannot feature extract mean amplitude without filtered data.')

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
            get_logger().error('Cannot feature extract signal negativity without filtered data.')

    def _calc_front_slope(self):
        if isinstance(self.filtered_data, pd.DataFrame):
            self.negative_slope = pd.DataFrame()
            for channel in self.filtered_data.columns:
                middle = len(self.filtered_data[channel]) // 2
                x1 = self.filtered_data[channel].iloc[:middle].idxmax()
                x2 = self.filtered_data[channel].iloc[x1+1:].idxmin()
                y = self.filtered_data[channel].iloc[:middle].max(), self.filtered_data[channel].iloc[x1+1:].min(),
                x = (x1, x2)
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                self.negative_slope[channel] = [slope]
        else:
            get_logger().error('Cannot feature extract negative slope without filtered data.')


    def extract_features(self):
        self._calc_negative_slope()
        self._calc_variability()
        self._calc_mean_amplitude()
        self._calc_signal_negativity()

    def get_features(self):
        existing_features = []
        if len(self.negative_slope) > 0:
            existing_features.append('negative_slope')
        if len(self.mean_amplitude) > 0:
            existing_features.append('mean_amplitude')
        if len(self.variability) > 0:
            existing_features.append('variability')
        if len(self.signal_negativity) > 0:
            existing_features.append('signal_negativity')

        return existing_features

    def plot(self, channel: int = 4, freq: int = 1200, show: bool = True, plot_features: bool = False,
             save_fig: bool = False,
             overwrite: bool=False) -> plt.figure():
        x_seconds = []
        fig = plt.figure(figsize=(5, 7))
        center = (len(self.data) / 2) / freq
        for i, row in self.filtered_data[channel].items():  # converts the window.data freqs to seconds
            x_seconds.append(i / freq - center)

        if not self.is_sub_window:
            if self.label == 1:
                agg_strat = self.aggregate_strategy

                emg_timestamp, tp_timestamp = self._timestamp_order(agg_strat)
                y_t = ['EC'] * len(tp_timestamp)
                y_t2 = ['EMG'] * len(emg_timestamp)

                gs = gridspec.GridSpec(ncols=1, nrows=6, figure=fig)
                ax1 = fig.add_subplot(gs[:2, 0])
                ax1.set_title(f' Channel: {channel + 1} - EEG {self.num_id + 1} - Raw  - Blink: {self.blink}')
                ax1.plot(x_seconds, self.data[channel], color='tomato')
                ax1.axvline(x=0, color='black', ls='--')

                ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
                ax2.set_title(f'Filter: {self.filter_type[channel].iloc[0]}')
                ax2.plot(x_seconds, self.filtered_data[channel], color='tomato', label='filtered data')
                ax2.axvline(x=0, color='black', ls='--')

                if plot_features:
                    x, y, slope = self._plot_features(center, freq, channel)
                    ax2.plot(x, y, label=f'slope {round(slope, 9)}', alpha=0.7)

                    mean = self.filtered_data[channel].mean()
                    y_mean = [mean] * len(self.filtered_data)
                    ax2.plot(x_seconds, y_mean, label=f'mean {round(self.filtered_data[channel].mean(), 7)}', alpha=0.7)

                    x_arr, y_arr, sum_negative = self._plot_signal_negativity(center, freq, channel)

                    ax2.plot(x_arr[0], y_arr[0], color='black',  label=f'negative {round(sum_negative, 7)}', alpha=1)
                    for xi, yi in zip(x_arr, y_arr):
                        ax2.plot(xi, yi, color='black',  alpha=1)

                    ax2.legend()

                ax3 = fig.add_subplot(gs[4, 0], sharex=ax1)
                ax3.set_title('EMG Detection')
                ax3.plot(emg_timestamp, y_t2, marker='^', color='limegreen')
                ax3.annotate('Peak', xy=[emg_timestamp[1], y_t2[1]])

                ax4 = fig.add_subplot(gs[5, 0], sharex=ax1)
                ax4.set_title('Execution Cue Interval')
                ax4.plot(tp_timestamp, y_t, marker='o', color='royalblue')
                ax4.annotate('Execution Cue', xy=[tp_timestamp[0], y_t[0]])

            else:
                gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.set_title(
                    f' Channel: {channel + 1} - EEG Window: {self.num_id + 1} - Raw - Blink: {self.blink}')
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

                    ax2.plot(x_arr[0], y_arr[0], color='black', label=f'negative', alpha=1)
                    for xi, yi in zip(x_arr, y_arr):
                        ax2.plot(xi, yi, color='black', alpha=1)
                    ax2.legend()
                ax2.axvline(x=0, color='black', ls='--')

            plt.tight_layout()

        if save_fig:
            if plot_features:
                path = f'{OUTPUT_PATH}/plots/window_plots/channel{channel}_{self.num_id}_feat.png'
            else:
                path = f'{OUTPUT_PATH}/plots/window_plots/channel{channel}_{self.num_id}.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=overwrite)
            except FileExistsError:
                get_logger().exception(f'Found file already exists: {file} you can '
                                       f'overwrite the file by setting overwrite=True')

            if show:
                plt.show()

        return fig

    def plot_window_for_all_channels(self, freq: int = 1200, save_fig: bool = False, overwrite: bool = False):
        # Finds a list of all EEG channels by checking their filter type
        eeg_channels = self.filter_type.apply(lambda row: row[row == 'bandpass'].index, axis=1)[0].tolist()
        fig = plt.figure(figsize=(14, 10))

        x_seconds = []
        center = (len(self.data) / 2) / freq

        for i, row in self.filtered_data[0].items():  # converts the window.data freqs to seconds
            x_seconds.append(i / freq - center)

        for channel in eeg_channels:
            ax = fig.add_subplot(3, 3, channel + 1)
            ax.set_title(f'Channel: {channel + 1}')
            ax.plot(x_seconds, self.filtered_data[channel], label='Filtered data')
            ax.axvline(x=0, color='black', ls='--')

        fig.suptitle(f'Window {self.num_id}', fontsize=16)
        plt.tight_layout()

        if save_fig:
            path = f'{OUTPUT_PATH}/plots/all_window_all_channel_plots/window{self.num_id}.png'
            file = os.path.split(path)[1]
            try:
                save_figure(path, fig, overwrite=overwrite)
            except FileExistsError:
                get_logger().exception(f'Found file already exists: {file} you can '
                                       f'overwrite the file by setting overwrite=True')

        plt.show()

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

    def _plot_features(self, center: int, freq: int, channel: int):
        x1, x2 = self.filtered_data[channel].idxmin(), self.filtered_data[channel].idxmax()
        y = self.filtered_data[channel].min(), self.filtered_data[channel].max()

        slope, intercept, r_value, p_value, std_err = linregress([x1, x2], y)
        return [x1 / freq - center, x2 / freq - center], y, slope


    def _plot_signal_negativity(self, center: int, freq: int, channel: int):

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
