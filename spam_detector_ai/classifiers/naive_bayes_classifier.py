# spam_detector_ai/classifiers/naive_bayes_classifier.py

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = None
        self.vectoriser = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_vectorized, y_train)

    def save_model(self, model_path, vectoriser_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.classifier, file)
        with open(vectoriser_path, 'wb') as file:
            pickle.dump(self.vectoriser, file)
