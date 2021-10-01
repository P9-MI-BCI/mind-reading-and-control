import pandas as pd


class Frame:

    def __int__(self, label: int = 0, data: pd.DataFrame = 0, timestamp: pd.Series = 0,
                filtered_data: pd.DataFrame = 0):
        self.label = label
        self.data = data
        self.timestamp = timestamp
        self.filtered_data = filtered_data

    def filter(self, filter_in, channel: int, **kwargs):
        self.filtered_data = pd.DataFrame(filter_in(self.data[channel], **kwargs))
