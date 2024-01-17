from time import time
import pickle
import os
import joblib


def load_with_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_with_joblib(file_path):
    return joblib.load(file_path)


file_name = "random_forest/random_forest_model.pkl"

# Load with pickle
t1 = time()
load_with_pickle(file_name)
print("Time for loading file size with pickle", os.path.getsize(file_name), "bytes =>", time() - t1)

# Load with joblib
t1 = time()
load_with_joblib(file_name)
print("Time for loading file size with joblib", os.path.getsize(file_name), "bytes =>", time() - t1)
