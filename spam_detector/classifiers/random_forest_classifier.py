# spam_detector/classifiers/random_forest_classifier.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from .base_classifier import BaseClassifier


class RandomForestSpamClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = None
        self.vectoriser = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
        self.smote = SMOTE(random_state=42)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        X_train_res, y_train_res = self.smote.fit_resample(X_train_vectorized, y_train)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        self.classifier.fit(X_train_res, y_train_res)

    def save_model(self, model_path, vectoriser_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.classifier, file)
        with open(vectoriser_path, 'wb') as file:
            pickle.dump(self.vectoriser, file)
