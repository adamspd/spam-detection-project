import os
import unittest

import pandas as pd

from spam_detector_ai.loading_and_processing.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    non_csv_file = None
    dummy_data_path = None

    @classmethod
    def setUpClass(cls):
        cls.dummy_data_path = "test_data.csv"
        pd.DataFrame({"text": ["sample text"]}).to_csv(cls.dummy_data_path, index=False)
        cls.non_csv_file = "test_data.txt"
        pd.DataFrame({"text": ["sample text"]}).to_string(cls.non_csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.dummy_data_path)
        os.remove(cls.non_csv_file)

    def test_load_data_successfully(self):
        """
        Data is loaded successfully.
        :return: None
        """
        data_loader = DataLoader(self.dummy_data_path)
        data = data_loader.get_data()
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)

    def test_file_not_found_error(self):
        """
        FileNotFoundError is raised when the file does not exist.
        :return: None
        """
        with self.assertRaises(FileNotFoundError):
            DataLoader("non_existent.csv")

    def test_non_csv_file_error(self):
        """
        ValueError is raised when the file is not a CSV file.
        :return: None
        """
        with self.assertRaises(ValueError):
            DataLoader(self.non_csv_file)


if __name__ == '__main__':
    unittest.main()
