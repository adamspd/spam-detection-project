import os

import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.prediction.predict import SpamDetector
from spam_detector_ai.training.train_models import ModelTrainer

CLASSIFIER_TYPES = [
    ClassifierType.NAIVE_BAYES,
    ClassifierType.RANDOM_FOREST,
    ClassifierType.SVM,
    ClassifierType.LOGISTIC_REGRESSION,
]
ACCURACY_THRESHOLD = 0.85


@pytest.fixture(scope="module")
def split_data():
    logger = init_logging()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_path = os.path.join(base_dir, 'data/spam.csv')
    trainer = ModelTrainer(data_path=data_path, logger=logger)
    processed_data = trainer.preprocess_data_()
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['processed_text'], processed_data['label'],
        test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def live_models(split_data):
    """Train each classifier from scratch so the test exercises the CURRENT
    training pipeline (classifier code + preprocessing + vectoriser), instead
    of whatever binary artefacts happen to be committed to the repo. This is
    what catches a regression in the code before it ships."""
    X_train, _, y_train, _ = split_data
    logger = init_logging()
    models = {}
    for ct in CLASSIFIER_TYPES:
        trainer = ModelTrainer(classifier_type=ct, logger=logger)
        trainer.classifier.train(X_train, y_train)
        models[ct] = trainer.classifier
    return models


class TestClassifiers:
    @pytest.mark.parametrize("ct", CLASSIFIER_TYPES, ids=lambda c: c.name)
    def test_live_pipeline_accuracy(self, ct, live_models, split_data):
        """The current training code must produce a model above threshold."""
        _, X_test, _, y_test = split_data
        classifier = live_models[ct]
        vectorized = classifier.vectoriser.transform(X_test).toarray()
        y_pred = classifier.classifier.predict(vectorized)
        acc = accuracy_score(y_test, y_pred)
        assert acc > ACCURACY_THRESHOLD, \
            f"{ct.name} live-pipeline accuracy {acc:.4f} <= {ACCURACY_THRESHOLD}"

    @pytest.mark.parametrize("ct", CLASSIFIER_TYPES, ids=lambda c: c.name)
    def test_committed_model_accuracy(self, ct, split_data):
        """The committed binaries that ship in the package must also be above
        threshold, so a stale/out-of-sync artefact can never quietly ship."""
        _, X_test, _, y_test = split_data
        detector = SpamDetector(model_type=ct)
        y_pred = [detector.test_is_spam(message) for message in X_test]
        acc = accuracy_score(y_test, y_pred)
        assert acc > ACCURACY_THRESHOLD, \
            f"{ct.name} committed-model accuracy {acc:.4f} <= {ACCURACY_THRESHOLD}"
