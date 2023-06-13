import os

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from classifier import ClassifierType
from logger_config import init_logging
from prediction import SpamDetector
from training import ModelTrainer


class TestModel:
    def __init__(self):
        self.classifier_types = [ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST, ClassifierType.SVM]
        self.logger = init_logging()

    def test_model(self, ct: ClassifierType):
        # Determine the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming the spam_detection directory is one level up from the current directory
        base_dir = os.path.dirname(current_dir)
        print(f"Base dir: {base_dir}")

        data_path = os.path.join(base_dir, 'data/spam.csv')

        trainer = ModelTrainer(data_path=data_path, classifier_type=ct, logger=self.logger)

        X_train, X_test, y_train, y_test = trainer._split_data()

        # Load the model using SpamDetector
        detector = SpamDetector(model_type=ct)

        # Predict on test data
        y_pred = [detector.test_is_spam(message) for message in X_test]

        # Output model type
        print(f"\nModel: {ct.name}")

        # Output the evaluation metrics
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

    def test(self):
        for classifier_type_ in self.classifier_types:
            self.test_model(ct=classifier_type_)

if __name__ == '__main__':
    tester = TestModel()
    tester.test()
