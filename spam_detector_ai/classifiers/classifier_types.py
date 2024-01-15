# spam_detector_ai/classifiers/classifier_types.py

from enum import Enum, auto


class ClassifierType(Enum):
    NAIVE_BAYES = auto()
    RANDOM_FOREST = auto()
    SVM = auto()
