import os
import pickle

from classifier.classifier_types import ClassifierType
from loading_and_processing import Preprocessor


class SpamDetector:
    def __init__(self, model_type=ClassifierType.NAIVE_BAYES):
        # Determine paths based on model type
        model_path, vectoriser_path = self.get_paths(model_type)

        # Load the saved classifier and vectoriser
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        with open(vectoriser_path, 'rb') as file:
            self.vectoriser = pickle.load(file)

        self.processor = Preprocessor()

    def get_paths(self, model_type):
        # Determine the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming the spam_detection directory is one level up from the current directory
        base_dir = os.path.dirname(current_dir)

        # Define the relative paths for each model type
        if model_type == ClassifierType.NAIVE_BAYES:
            relative_path_model = 'models/bayes/naive_bayes_model.pkl'
            relative_path_vectoriser = 'models/bayes/naive_bayes_vectoriser.pkl'
        elif model_type == ClassifierType.RANDOM_FOREST:
            relative_path_model = 'models/random_forest/random_forest_model.pkl'
            relative_path_vectoriser = 'models/random_forest/random_forest_vectoriser.pkl'
        elif model_type == ClassifierType.SVM:
            relative_path_model = 'models/svm/svm_model.pkl'
            relative_path_vectoriser = 'models/svm/svm_vectoriser.pkl'
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Construct the absolute paths by joining the base directory with the relative paths
        absolute_path_model = os.path.join(base_dir, relative_path_model)
        absolute_path_vectoriser = os.path.join(base_dir, relative_path_vectoriser)

        return absolute_path_model, absolute_path_vectoriser

    def is_spam(self, message_):
        # Preprocess the message
        processed_message = self.processor.preprocess_text(message_)

        # Vectorize the preprocessed message
        vectorized_message = self.vectoriser.transform([processed_message]).toarray()

        # Make prediction
        prediction = self.model.predict(vectorized_message)

        # Return True if spam, False if not spam
        return prediction[0] == 'spam'

    def test_is_spam(self, message_):
        processed_message = self.processor.preprocess_text(message_)

        # Vectorize the preprocessed message
        vectorized_message = self.vectoriser.transform([processed_message]).toarray()

        # Make prediction
        prediction = self.model.predict(vectorized_message)

        # Return True if spam, False if not spam
        return prediction[0]


class VotingSpamDetector:
    def __init__(self):
        self.detectors = [
            SpamDetector(model_type=ClassifierType.NAIVE_BAYES),
            SpamDetector(model_type=ClassifierType.RANDOM_FOREST),
            SpamDetector(model_type=ClassifierType.SVM)
        ]

    def is_spam(self, message_):
        # Count the number of spam predictions
        spam_votes = sum(detector.is_spam(message_) for detector in self.detectors)
        print(f"spam_votes: {[detector.is_spam(message_) for detector in self.detectors]}")
        # Count the number of ham (not spam) predictions
        ham_votes = len(self.detectors) - spam_votes
        # Majority voting: if the majority of detectors say it's spam, return True, otherwise False
        return spam_votes > ham_votes


if __name__ == "__main__":
    voting_detector = VotingSpamDetector()

    message = "Hi John, I hope you are doing well. I wanted to follow up on the meeting we had last week. When can we schedule the next meeting? Best, Jane"
    print("Voting -> Is spam:", voting_detector.is_spam(message))
