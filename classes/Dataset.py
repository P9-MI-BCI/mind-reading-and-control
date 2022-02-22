#from utility.save_figure import save_figure
from utility.logger import get_logger
from definitions import OUTPUT_PATH
import matplotlib.pyplot as plt
import os
import pandas as pd


class Dataset:

    def __init__(self, sample_rate=0, data=pd.DataFrame, filtered_data=pd.DataFrame, label=999, onsets_index=0):
        self.sample_rate = sample_rate
        self.data = data
        self.filtered_data = filtered_data  # experimental be careful of use
        self.label = label
        self.onsets_index = onsets_index

