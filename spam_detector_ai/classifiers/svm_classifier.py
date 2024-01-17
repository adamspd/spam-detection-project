# spam_detector_ai/classifiers/svm_classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        self.classifier = SVC(C=10, gamma=1, kernel='rbf')
        self.classifier.fit(X_train_vectorized, y_train)
