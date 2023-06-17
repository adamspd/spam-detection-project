# spam_detector_ai/trainer.py

from sklearn.model_selection import train_test_split

from classifiers.classifier_types import ClassifierType
from logger_config import init_logging
from training import ModelTrainer

logger = init_logging()


def train_model(classifier_type, model_filename, vectoriser_filename, X_train, y_train):
    logger.info(f'\nTraining {classifier_type}')
    trainer_ = ModelTrainer(data=None, classifier_type=classifier_type, logger=logger)
    trainer_.train(X_train, y_train)
    trainer_.save_model(model_filename, vectoriser_filename)


if __name__ == '__main__':
    # Load and preprocess data once
    initial_trainer = ModelTrainer(data_path='data/spam.csv', classifier_type=None, logger=logger)
    processed_data = initial_trainer._preprocess_data()

    # Split the data once
    X__train, _, y__train, _ = train_test_split(processed_data['processed_text'], processed_data['label'], test_size=0.2, random_state=0)

    # Configurations for each model
    configurations = [
        (ClassifierType.SVM, 'svm_model.pkl', 'svm_vectoriser.pkl'),
        (ClassifierType.NAIVE_BAYES, 'naive_bayes_model.pkl', 'naive_bayes_vectoriser.pkl'),
        (ClassifierType.RANDOM_FOREST, 'random_forest_model.pkl', 'random_forest_vectoriser.pkl')
    ]

    # Train each model with the pre-split data
    for ct, mf, vf in configurations:
        train_model(ct, mf, vf, X__train, y__train)

