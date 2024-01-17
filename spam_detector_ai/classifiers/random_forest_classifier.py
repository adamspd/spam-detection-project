# spam_detector_ai/classifiers/random_forest_classifier.py

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_classifier import BaseClassifier


class RandomForestSpamClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.smote = SMOTE(random_state=42)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        X_train_res, y_train_res = self.smote.fit_resample(X_train_vectorized, y_train)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        self.classifier.fit(X_train_res, y_train_res)
