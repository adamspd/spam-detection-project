# trainer.py

from classifier.classifier_types import ClassifierType
from logger_config import init_logging

from training import ModelTrainer

logger = init_logging()


def train_naive_bayes():
    # Train NAIVE_BAYES
    logger.info('Training Naive Bayes')
    trainer_ = ModelTrainer(data_path='data/spam.csv', classifier_type=ClassifierType.NAIVE_BAYES, logger=logger)
    trainer_.train()
    trainer_.save_model('naive_bayes_model.pkl', 'naive_bayes_vectoriser.pkl')


def train_random_forest():
    # Train RANDOM_FOREST
    logger.info('Training Random Forest')
    trainer_ = ModelTrainer(data_path='data/spam.csv', classifier_type=ClassifierType.RANDOM_FOREST, logger=logger)
    trainer_.train()
    trainer_.save_model('random_forest_model.pkl', 'random_forest_vectoriser.pkl')


def train_svm():
    # Train SVM
    logger.info('Training SVM')
    trainer_ = ModelTrainer(data_path='data/spam.csv', classifier_type=ClassifierType.SVM, logger=logger)
    trainer_.train()
    trainer_.save_model('svm_model.pkl', 'svm_vectoriser.pkl')


if __name__ == '__main__':
    train_svm()
    train_naive_bayes()
    train_random_forest()
