# spam_detector_ai/training/train_models.py

import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.classifiers.logistic_regression_classifier import LogisticRegressionSpamClassifier
from spam_detector_ai.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from spam_detector_ai.classifiers.random_forest_classifier import RandomForestSpamClassifier
from spam_detector_ai.classifiers.svm_classifier import SVMClassifier
from spam_detector_ai.classifiers.xgb_classifier import XGBSpamClassifier
from spam_detector_ai.loading_and_processing.data_loader import DataLoader
from spam_detector_ai.loading_and_processing.preprocessor import Preprocessor


class ModelTrainer:
    def __init__(self, data_path=None, data=None, classifier_type=ClassifierType.NAIVE_BAYES, test_size=0.2,
                 logger=None):
        self.data_path = data_path
        self.classifier_type = classifier_type
        self.test_size = test_size
        self.data = data
        self.processed_data = None
        self.logger = logger
        self.logger.info(f'ModelTrainer initialized with classifier type: {classifier_type}')
        self.classifier = self.get_classifier_(classifier_type)

    def preprocess_data_(self):
        self.logger.info('Preprocessing data')
        if self.data is None:
            self.logger.info(f'Loading data from {self.data_path}')
            self.data = DataLoader(self.data_path).get_data()
        self.processed_data = Preprocessor().preprocess(self.data)
        return self.processed_data

    def split_data_(self):
        self.logger.info('Splitting data')
        if self.processed_data is None:
            self.processed_data = self.preprocess_data_()
        return train_test_split(self.processed_data['processed_text'], self.processed_data['label'],
                                test_size=self.test_size, random_state=0)

    def get_classifier_(self, classifier_type):
        classifier_map = {
            ClassifierType.NAIVE_BAYES.value: NaiveBayesClassifier(),
            ClassifierType.RANDOM_FOREST.value: RandomForestSpamClassifier(),
            ClassifierType.SVM.value: SVMClassifier(),
            ClassifierType.XGB.value: XGBSpamClassifier(),
            ClassifierType.LOGISTIC_REGRESSION.value: LogisticRegressionSpamClassifier(),
        }
        classifier = classifier_map.get(classifier_type.value)
        if classifier:
            return classifier
        else:
            self.logger.error(f"Invalid classifier type: {classifier_type}")
            raise ValueError(f"Invalid classifier type: {classifier_type}")

    def train(self, X_train, y_train):
        self.logger.info('Training started.')

        self.classifier.train(X_train, y_train)
        self.logger.info('Training completed.')

    def get_directory_path(self):
        directory_map = {
            ClassifierType.NAIVE_BAYES.value: 'models/bayes',
            ClassifierType.RANDOM_FOREST.value: 'models/random_forest',
            ClassifierType.SVM.value: 'models/svm',
            ClassifierType.XGB.value: 'models/xgb',
            ClassifierType.LOGISTIC_REGRESSION.value: 'models/logistic_regression'
        }
        directory_path = directory_map.get(self.classifier_type.value)
        if directory_path:
            return directory_path
        else:
            raise ValueError(f"Invalid classifier type: {self.classifier_type}")

    def save_model(self, model_filename, vectoriser_filename):
        # Use the project root to construct the paths
        project_root = Path(__file__).parent.parent
        models_dir = project_root
        directory_path = self.get_directory_path()

        model_filepath = models_dir / directory_path / model_filename
        vectoriser_filepath = models_dir / directory_path / vectoriser_filename

        # Ensure the directory exists
        model_filepath.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Saving model to {model_filepath}')
        self.classifier.save_model(str(model_filepath), str(vectoriser_filepath))
        self.logger.info('Model saved.\n')
