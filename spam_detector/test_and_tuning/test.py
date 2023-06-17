# spam_detector/test_and_tuning/test.py

import os

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from spam_detector.classifiers.classifier_types import ClassifierType
from spam_detector.logger_config import init_logging
from spam_detector.prediction import SpamDetector
from spam_detector.training import ModelTrainer


class TestModel:
    def __init__(self):
        self.classifier_types = [ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST, ClassifierType.SVM]
        self.logger = init_logging()
        # Determine the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming the spam_detector directory is one level up from the current directory
        base_dir = os.path.dirname(current_dir)
        data_path = os.path.join(base_dir, 'data/spam.csv')
        self.initial_trainer = ModelTrainer(data_path=data_path, classifier_type=None, logger=self.logger)
        self.processed_data = self.initial_trainer._preprocess_data()

        # Split the data once
        _, self.X_test, _, self.y_test = train_test_split(self.processed_data['processed_text'], self.processed_data['label'],
                                                            test_size=0.2, random_state=0)

    def test_model(self, ct: ClassifierType):
        # Load the model using SpamDetector
        detector = SpamDetector(model_type=ct)

        # Predict on test data
        y_pred = [detector.test_is_spam(message) for message in self.X_test]

        # Output model type
        print(f"\nModel: {ct.name}")

        # Output the evaluation metrics
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))
        print(accuracy_score(self.y_test, y_pred))

    def test(self):
        for classifier_type_ in self.classifier_types:
            self.test_model(ct=classifier_type_)

if __name__ == '__main__':
    tester = TestModel()
    tester.test()
