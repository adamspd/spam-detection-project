# data_loader.py

import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def get_data(self):
        return self.data
