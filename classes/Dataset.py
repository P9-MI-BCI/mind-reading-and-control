import pandas as pd

class Dataset:

    def __init__(self, sample_rate=0, data=pd.DataFrame(), filtered_data=pd.DataFrame(), label=999, clusters=0,
                 filename='filename_goes_here'):
        self.sample_rate = sample_rate
        self.data = data
        self.filtered_data = filtered_data  # experimental be careful of use
        self.label = label
        self.clusters = clusters
        self.filename = filename
