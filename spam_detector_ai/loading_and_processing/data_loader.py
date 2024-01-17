# spam_detector_ai/loading_and_processing/data_loader.py

import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        if not data_path.endswith('.csv'):
            raise ValueError("Only CSV files are supported")
        try:
            self.data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {data_path} was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the file: {e}")

    def get_data(self):
        """
        Return the loaded data.
        """
        return self.data
