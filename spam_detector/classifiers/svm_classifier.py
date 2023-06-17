# spam_detector/classifiers/svm_classifier.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = None
        self.vectoriser = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        self.classifier = SVC(C=100, gamma=1, kernel='rbf')
        self.classifier.fit(X_train_vectorized, y_train)

    def save_model(self, model_path, vectoriser_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.classifier, file)
        with open(vectoriser_path, 'wb') as file:
            pickle.dump(self.vectoriser, file)
