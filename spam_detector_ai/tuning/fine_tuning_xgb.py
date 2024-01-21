from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from spam_detector_ai.classifiers.base_classifier import BaseClassifier
from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.training.train_models import ModelTrainer

if __name__ == '__main__':
    logger = init_logging()

    logger.info("Define the parameter grid")
    param_grid_xgb = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1]
    }

    logger.info("Define the grid search")
    grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, refit=True, verbose=3, cv=5, n_jobs=-1)

    logger.info("Loading the training data")
    trainer = ModelTrainer(data_path='../data/spam.csv', classifier_type=ClassifierType.XGB, logger=logger)

    logger.info("Splitting the data")
    X_train, X_test, y_train, y_test = trainer.split_data_()

    # Vectorize the data
    logger.info("Vectorising the data")
    vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
    X_train_vect = vectoriser.fit_transform(X_train)
    X_test_vect = vectoriser.transform(X_test)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logger.info("Searching for the best parameters")
    grid_search_xgb.fit(X_train_vect, y_train_encoded)

    # Get the best parameters and model
    best_params_xgb = grid_search_xgb.best_params_
    logger.info(f"Best Parameters: {best_params_xgb}")

    best_estimator_xgb = grid_search_xgb.best_estimator_

    # Predict using the best model
    logger.info("Predicting using the best model")
    y_pred = best_estimator_xgb.predict(X_test_vect)

    # Output the evaluation metrics
    logger.info("Outputting the evaluation metrics")
    print(confusion_matrix(y_test_encoded, y_pred))
    print(classification_report(y_test_encoded, y_pred))
    print(accuracy_score(y_test_encoded, y_pred))
