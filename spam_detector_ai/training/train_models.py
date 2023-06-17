# spam_detector_ai/training/train_models.py

import os

from sklearn.model_selection import train_test_split

from spam_detector_ai.classifiers import RandomForestSpamClassifier, NaiveBayesClassifier, SVMClassifier, ClassifierType
from spam_detector_ai.loading_and_processing import DataLoader, Preprocessor


class ModelTrainer:
    def __init__(self, data_path=None, data=None, classifier_type=None, test_size=0.2, logger=None):
        self.data_path = data_path
        self.classifier_type = classifier_type
        self.test_size = test_size
        self.data = data
        self.processed_data = None
        self.logger = logger
        self.logger.info(f'ModelTrainer initialized with classifier type: {classifier_type}')
        if classifier_type is not None:
            self.classifier = self._get_classifier(classifier_type)

    def _preprocess_data(self):
        self.logger.info('Preprocessing data')
        if self.data is None:
            self.logger.info(f'Loading data from {self.data_path}')
            self.data = DataLoader(self.data_path).get_data()
        self.processed_data = Preprocessor().preprocess(self.data)
        return self.processed_data

    def _split_data(self):
        self.logger.info('Splitting data')
        if self.processed_data is None:
            self.processed_data = self._preprocess_data()
        return train_test_split(self.processed_data['processed_text'], self.processed_data['label'],
                                test_size=self.test_size, random_state=0)

    def _get_classifier(self, classifier_type):
        classifier_map = {
            ClassifierType.NAIVE_BAYES.value: NaiveBayesClassifier(),
            ClassifierType.RANDOM_FOREST.value: RandomForestSpamClassifier(),
            ClassifierType.SVM.value: SVMClassifier()
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
            ClassifierType.SVM.value: 'models/svm'
        }
        directory_path = directory_map.get(self.classifier_type.value)
        if directory_path:
            return directory_path
        else:
            raise ValueError(f"Invalid classifier type: {self.classifier_type}")

    def save_model(self, model_filename, vectoriser_filename):
        # Determine the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming the spam_detector_ai directory is one level up from the current directory
        base_dir = os.path.dirname(current_dir)
        directory_path = self.get_directory_path()

        # Ensure the directory exists
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        model_filepath = os.path.join(base_dir, directory_path, model_filename)
        vectoriser_filepath = os.path.join(base_dir, directory_path, vectoriser_filename)

        self.logger.info(f'Saving model to {model_filepath}')
        self.classifier.save_model(model_filepath, vectoriser_filepath)
        self.logger.info('Model saved.')
