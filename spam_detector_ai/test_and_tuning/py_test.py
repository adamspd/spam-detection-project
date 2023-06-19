import os
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.prediction import SpamDetector
from spam_detector_ai.training import ModelTrainer


@pytest.fixture(scope="module")
def test_model():
    classifier_types = [ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST, ClassifierType.SVM]
    logger = init_logging()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_path = os.path.join(base_dir, 'data/spam.csv')
    initial_trainer = ModelTrainer(data_path=data_path, classifier_type=None, logger=logger)
    processed_data = initial_trainer._preprocess_data()
    _, X_test, _, y_test = train_test_split(processed_data['processed_text'], processed_data['label'],
                                            test_size=0.2, random_state=0)
    return classifier_types, X_test, y_test


class TestClassifiers:
    def test_classifier_accuracy(self, test_model):
        classifier_types, X_test, y_test = test_model
        for ct in classifier_types:
            detector = SpamDetector(model_type=ct)
            y_pred = [detector.test_is_spam(message) for message in X_test]
            assert accuracy_score(y_test, y_pred) > 0.85
