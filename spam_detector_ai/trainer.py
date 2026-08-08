# spam_detector_ai/trainer.py
"""Train all 5 spam detection models and save them to the models/ directory."""
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.training.train_models import ModelTrainer

logger = init_logging()


def train_model(classifier_type, model_filename, vectoriser_filename, X_train, y_train):
    logger.info(f'Training {classifier_type}')
    trainer_ = ModelTrainer(data=None, classifier_type=classifier_type, logger=logger)
    trainer_.train(X_train, y_train)
    trainer_.save_model(model_filename, vectoriser_filename)


if __name__ == '__main__':
    # Load and preprocess data once
    # The current file is at spam_detector_ai/trainer.py, so project root is one level up
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'spam_detector_ai' / 'data' / 'spam.csv'
    initial_trainer = ModelTrainer(data_path=str(data_path), logger=logger)
    processed_data = initial_trainer.preprocess_data_()

    # Split the data once
    X__train, _, y__train, _ = train_test_split(processed_data['processed_text'], processed_data['label'],
                                                test_size=0.2, random_state=0)

    # Configurations for each model
    configurations = [
        (ClassifierType.SVM, 'svm_model.joblib', 'svm_vectoriser.joblib'),
        (ClassifierType.NAIVE_BAYES, 'naive_bayes_model.joblib', 'naive_bayes_vectoriser.joblib'),
        (ClassifierType.RANDOM_FOREST, 'random_forest_model.joblib', 'random_forest_vectoriser.joblib'),
        (ClassifierType.XGB, 'xgb_model.json', 'xgb_vectoriser.joblib'),
        (ClassifierType.LOGISTIC_REGRESSION, 'logistic_regression_model.joblib',
         'logistic_regression_vectoriser.joblib')
    ]

    # Train each model with the pre-split data
    logger.info(f"Train each model with the pre-split data\n")
    for ct, mf, vf in configurations:
        train_model(ct, mf, vf, X__train, y__train)
