# spam_detector_ai/classifiers/naive_bayes_classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    # Naive Bayes is the weakest model in the suite and historically sat right on
    # the 0.85 test threshold. TF-IDF features (with bigrams) plus ComplementNB
    # - which is designed for imbalanced text like this 65/35 ham/spam set -
    # give a durable margin above the threshold.
    VECTORIZER_PARAMS = {
        'max_features': 7000,
        'min_df': 2,
        'max_df': 0.9,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
    }

    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**NaiveBayesClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        self.classifier = ComplementNB()
        self.classifier.fit(X_train_vectorized, y_train)
