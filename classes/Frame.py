import pandas as pd


class Frame:

    def __int__(self, label: int = 0, data: pd.DataFrame = 0, timestamp: pd.Series = 0,
                filtered_data: pd.DataFrame = 0, filter_type: pd.DataFrame = 0):
        self.label = label
        self.data = data
        self.timestamp = timestamp
        self.filtered_data = filtered_data
        self.filter_type = 0

    def filter(self, filter_in, channel: int, **kwargs):
        try:
            try:
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.filtered_data[channel], **kwargs))
            except KeyError:
                self.filtered_data[channel] = pd.DataFrame(filter_in(self.data[channel], **kwargs))
        except AttributeError:
            self.filtered_data = pd.DataFrame(filter_in(self.data[channel], **kwargs))

    def update_filter_type(self, filter_types: pd.DataFrame):
        for channel in filter_types:
            try:
                try:
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
                except KeyError:
                    self.filter_type[channel] = [filter_types[channel].iloc[0]]
            except AttributeError:
                self.filter_type = pd.DataFrame()
                self.filter_type[channel] = [filter_types[channel].iloc[0]]
