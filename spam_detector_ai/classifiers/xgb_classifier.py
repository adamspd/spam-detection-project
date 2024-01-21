# spam_detector_ai/classifiers/xgb_classifier.py
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_classifier import BaseClassifier


class XGBSpamClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.label_encoder = LabelEncoder()

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.classifier = XGBClassifier(colsample_bytree=0.8, learning_rate=0.2, max_depth=5, n_estimators=300,
                                        subsample=1)
        self.classifier.fit(X_train_vectorized, y_train_encoded)

    def predict(self, X_test):
        X_test_vectorized = self.vectoriser.transform(X_test)
        predictions = self.classifier.predict(X_test_vectorized)
        return self.label_encoder.inverse_transform(predictions)
