from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from spam_detector_ai.classifiers.base_classifier import BaseClassifier
from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.training.train_models import ModelTrainer

if __name__ == '__main__':
    logger = init_logging()

    logger.info("Define the parameter grid for Logistic Regression")
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],  # These solvers support l1 penalty
        'max_iter': [200, 500, 1000]  # Increase max_iter
    }

    logger.info("Define the grid search for Logistic Regression")
    grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, refit=True, verbose=3, cv=5, n_jobs=-1)

    logger.info("Loading the training data")
    trainer = ModelTrainer(data_path='../data/spam.csv', classifier_type=ClassifierType.LOGISTIC_REGRESSION,
                           logger=logger)

    logger.info("Splitting the data")
    X_train, X_test, y_train, y_test = trainer.split_data_()

    # Vectorize the data
    logger.info("Vectorising the data")
    vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
    X_train_vect = vectoriser.fit_transform(X_train)
    X_test_vect = vectoriser.transform(X_test)

    # Perform the grid search
    grid_search_lr.fit(X_train_vect, y_train)

    # Get the best parameters and model
    best_params_lr = grid_search_lr.best_params_
    logger.info(f"Best Parameters for Logistic Regression: {best_params_lr}")

    best_estimator_lr = grid_search_lr.best_estimator_

    # Predict using the best model
    logger.info("Predicting using the best model for Logistic Regression")
    y_pred_lr = best_estimator_lr.predict(X_test_vect)

    # Output the evaluation metrics
    logger.info("Outputting the evaluation metrics for Logistic Regression")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    print(accuracy_score(y_test, y_pred_lr))
