# spam_detector_ai/test_and_tuning/fine_tuning_svm.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.training.train_models import ModelTrainer

if __name__ == '__main__':
    logger = init_logging()

    logger.info("Define the parameter grid")
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    logger.info("Define the grid search")
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)

    logger.info("Loading the training data")
    trainer = ModelTrainer(data_path='../data/spam.csv', classifier_type=ClassifierType.SVM, logger=logger)

    logger.info("Splitting the data")
    X_train, X_test, y_train, y_test = trainer.split_data_()

    logger.info("Vectorise the data")
    vectoriser = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
    X_train_vect = vectoriser.fit_transform(X_train)
    X_test_vect = vectoriser.transform(X_test)

    logger.info("Searching for the best parameters")
    grid_search.fit(X_train_vect, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    logger.info(f"Best Parameters: {best_params}")

    logger.info("Getting the best estimator")
    best_estimator = grid_search.best_estimator_

    # Predict using the best model
    logger.info("Predicting using the best model")
    y_pred = best_estimator.predict(X_test_vect)

    # Output the evaluation metrics
    logger.info("Outputting the evaluation metrics")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
