# spam_detector_ai/prediction/predict.py
"""
Author: Adams P. David
Contact: https://adamspierredavid.com/contact/
Date Written: 2023-06-12
"""

import os
import sys
from pathlib import Path

# Add the project root to the path if running as a script
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.classifiers.logistic_regression_classifier import LogisticRegressionSpamClassifier
from spam_detector_ai.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from spam_detector_ai.classifiers.random_forest_classifier import RandomForestSpamClassifier
from spam_detector_ai.classifiers.svm_classifier import SVMClassifier
from spam_detector_ai.classifiers.xgb_classifier import XGBSpamClassifier
from spam_detector_ai.loading_and_processing.preprocessor import Preprocessor
from spam_detector_ai.prediction.performance import ModelAccuracy


def get_model_path(model_type):
    # Determine the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the spam_detector_ai directory is one level up from the current directory
    base_dir = os.path.dirname(current_dir)

    # Define the relative paths for each model type using a dictionary
    paths_map = {
        ClassifierType.NAIVE_BAYES: (
            'models/bayes/naive_bayes_model.joblib',
            'models/bayes/naive_bayes_vectoriser.joblib'
        ),
        ClassifierType.RANDOM_FOREST: (
            'models/random_forest/random_forest_model.joblib',
            'models/random_forest/random_forest_vectoriser.joblib'
        ),
        ClassifierType.SVM: (
            'models/svm/svm_model.joblib',
            'models/svm/svm_vectoriser.joblib'
        ),
        ClassifierType.XGB: (
            'models/xgb/xgb_model.json',
            'models/xgb/xgb_vectoriser.joblib'
        ),
        ClassifierType.LOGISTIC_REGRESSION: (
            'models/logistic_regression/logistic_regression_model.joblib',
            'models/logistic_regression/logistic_regression_vectoriser.joblib'
        )
    }

    relative_path_model, relative_path_vectoriser = paths_map.get(model_type)

    if relative_path_model and relative_path_vectoriser:
        # Construct the absolute paths by joining the base directory with the relative paths
        absolute_path_model = os.path.join(base_dir, relative_path_model)
        absolute_path_vectoriser = os.path.join(base_dir, relative_path_vectoriser)
        return absolute_path_model, absolute_path_vectoriser
    else:
        raise ValueError(f"Invalid model type: {model_type}")


class SpamDetector:
    """This class is used to detect whether a message is spam or not spam."""

    def __init__(self, model_type=ClassifierType.NAIVE_BAYES):
        classifier_map = {
            ClassifierType.NAIVE_BAYES.value: NaiveBayesClassifier(),
            ClassifierType.RANDOM_FOREST.value: RandomForestSpamClassifier(),
            ClassifierType.SVM.value: SVMClassifier(),
            ClassifierType.XGB.value: XGBSpamClassifier(),
            ClassifierType.LOGISTIC_REGRESSION.value: LogisticRegressionSpamClassifier(),
        }
        classifier = classifier_map.get(model_type.value)
        if not classifier:
            raise ValueError(f"Invalid model type: {model_type}")

        self.model = classifier
        model_path, vectoriser_path = get_model_path(model_type)
        self.model.load_model(model_path, vectoriser_path)
        self.processor = Preprocessor()

    def is_spam(self, message_):
        # Preprocess the message
        processed_message = self.processor.preprocess_text(message_)

        # Vectorize the preprocessed message
        vectorized_message = self.model.vectoriser.transform([processed_message]).toarray()

        # Make prediction
        prediction = self.model.classifier.predict(vectorized_message)

        # Return True if spam, False if not spam
        return prediction[0] == 'spam'

    def test_is_spam(self, message_):
        processed_message = self.processor.preprocess_text(message_)

        # Vectorize the preprocessed message
        vectorized_message = self.model.vectoriser.transform([processed_message]).toarray()

        # Make prediction
        prediction = self.model.classifier.predict(vectorized_message)

        # Return True if spam, False if not spam
        return prediction[0]


class VotingSpamDetector:
    """This class is used to detect whether a message is spam
    or not spam using majority voting of multiple spam detectors models."""

    def __init__(self):
        total_accuracy = ModelAccuracy.total_accuracy()
        self.detectors = [
            (SpamDetector(model_type=ClassifierType.NAIVE_BAYES), ModelAccuracy.NAIVE_BAYES / total_accuracy),
            (SpamDetector(model_type=ClassifierType.RANDOM_FOREST), ModelAccuracy.RANDOM_FOREST / total_accuracy),
            (SpamDetector(model_type=ClassifierType.SVM), ModelAccuracy.SVM / total_accuracy),
            (SpamDetector(model_type=ClassifierType.LOGISTIC_REGRESSION), ModelAccuracy.LOGISTIC_REG / total_accuracy),
            (SpamDetector(model_type=ClassifierType.XGB), ModelAccuracy.XGB / total_accuracy)
        ]

    def is_spam(self, message_):
        total_weight = sum(weight for _, weight in self.detectors)
        decision_threshold = total_weight / 2
        votes = [(detector.is_spam(message_), weight) for detector, weight in self.detectors]
        weighted_spam_score = sum(vote * weight for vote, weight in votes)

        # Interpret and display the voting results
        vote_descriptions = [f"{'Spam' if vote else 'Ham'} (Weight: {weight:.4f})" for vote, weight in votes]
        decision = "Spam" if weighted_spam_score > 0.50 else "Ham"
        print(f"Votes: {vote_descriptions}, Weighted Spam Score: {weighted_spam_score:.4f}, Classified as: {decision}")

        return weighted_spam_score > decision_threshold


if __name__ == "__main__":
    voting_detector = VotingSpamDetector()

    message_1 = "Hello!"
    print("Message 1 -> Is spam:", voting_detector.is_spam(message_1), f"Expected: True")
    message_2 = (f"Hi, I noticed your website hasn't embraced AI capabilities yet. Would you be interested in a "
                 f"suggestion I have?")
    print("Message 2 -> Is spam:", voting_detector.is_spam(message_2), f"Expected: True")
    message_3 = (f"Developed by a Construction Specific CPA Firm, TimeSuite is the worlds most advanced Construction "
                 f"Software. TimeSuite is next generation. Advanced because it’s intuitive, comprehensive and "
                 f"dynamic. Advanced because it’s has a relational architecture (no modular subsystems/no modules). "
                 f"Web, desktop and mobile interfaces to a single database. One system, 3 comprehensive interfaces. "
                 f"Project Management, Accounting, Scheduling, Estimating, On-Screen Take Off, PDF Viewer, "
                 f"CAD Drawing Layering, Geo Timecards, CRM, Task Management, Resource Management, Banking System "
                 f"Integration, Text Messaging, Email, Calendar, Form Creation, Property Management, "
                 f"RFQs/Bid Packages, Outlook and Google email and calendar integrations and more. Fully automated "
                 f"percentage of completion method of accounting with a full job schedule that always ties to the "
                 f"income statement. Gain access to a live fully functional demo at TimeSuite.com.")
    print("Message 3 -> Is spam:", voting_detector.is_spam(message_3), f"Expected: True")
    message_4 = (f"Bonsoir mwen se Haitian mwen bezwen pran seminaire sou profesyon siw fè please retounenm poum pran "
                 f"kontak avem mesi.")
    print("Message 4 -> Is spam:", voting_detector.is_spam(message_4), f"Expected: False")
    message_5 = (f"subject: Want a birthday shoot for my little brother and I. I am one of Jeff and Germina's"
                 f"friends and Germina' friends they give me your website and our birthday is on the 21st of "
                 f"February please lemme know.")
    print("Message 5 -> Is spam:", voting_detector.is_spam(message_5), f"Expected: False")
    message_6 = f"Hello, I went to your blog and I really like your article. Thanks for doing such a great job."
    print("Message 6 -> Is spam:", voting_detector.is_spam(message_6), f"Expected: False")
    message_7 = f"Hi, when I tried browsing through your article, I found a few broken links. Please fix them."
    print("Message 7 -> Is spam:", voting_detector.is_spam(message_7), f"Expected: False")
