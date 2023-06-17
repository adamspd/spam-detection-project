# spam_detector/classifiers/base_classifier.py

from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def save_model(self, model_path, vectoriser_path):
        pass
