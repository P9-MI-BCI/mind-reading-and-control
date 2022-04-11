import numpy as np


class Cluster:

    def __init__(self, data=[]):
        self.data = np.array(data)
        self.peak = 0
        self.start = 0
        self.end = 0
        self.length = 0



    def create_info(self):
        self.data.sort()
        self.start = self.data[0]
        self.end = self.data[-1]
        self.length = self.end - self.start