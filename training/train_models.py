# train_models.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from classifier import RandomForestSpamClassifier, NaiveBayesClassifier, SVMClassifier, ClassifierType

from loading_and_processing import DataLoader, Preprocessor


class ModelTrainer:
    def __init__(self, data_path, classifier_type, test_size=0.2, logger=None):
        self.data_path = data_path
        self.classifier_type = classifier_type
        self.test_size = test_size
        self.data = None
        self.processed_data = None
        self.logger = logger
        self.logger.info(f'ModelTrainer initialized with classifier type: {classifier_type}')
        self.classifier = self._get_classifier(classifier_type)  # Move this line below self.logger = logger

    def _load_data(self):
        self.logger.info(f'Loading data from {self.data_path}')
        data_loader = DataLoader(self.data_path)
        self.data = data_loader.get_data()
        return self.data

    def _preprocess_data(self, data=None):
        self.logger.info('Preprocessing data')
        if data is None:
            if self.data is None:
                self.data = self._load_data()
            data = self.data

        preprocessor = Preprocessor()
        self.processed_data = preprocessor.preprocess(data)
        return self.processed_data

    def _split_data(self, data=None):
        self.logger.info('Splitting data')
        if data is None:
            if self.processed_data is None:
                self.processed_data = self._preprocess_data()
            data = self.processed_data

        return train_test_split(data['processed_text'], data['label'], test_size=self.test_size, random_state=0)

    def _vectorize_data(self, X_train, X_test):
        vectoriser = TfidfVectorizer()
        X_train_vect = vectoriser.fit_transform(X_train)
        X_test_vect = vectoriser.transform(X_test)
        return X_train_vect, X_test_vect

    def _get_classifier(self, classifier_type):
        if classifier_type is ClassifierType.NAIVE_BAYES:
            return NaiveBayesClassifier()
        elif classifier_type is ClassifierType.RANDOM_FOREST:
            return RandomForestSpamClassifier()
        elif classifier_type is ClassifierType.SVM:
            return SVMClassifier()
        else:
            self.logger.error(f"Invalid classifier type: {classifier_type}")
            raise ValueError(f"Invalid classifier type: {classifier_type}")

    def train(self):
        self.logger.info('Training started.')
        data = self._load_data()
        data = self._preprocess_data(data)
        X_train, _, y_train, _ = self._split_data(data)
        self.classifier.train(X_train, y_train)
        self.logger.info('Training completed.')

    def get_directory_path(self):
        if self.classifier_type == ClassifierType.NAIVE_BAYES:
            return 'models/bayes'
        elif self.classifier_type == ClassifierType.RANDOM_FOREST:
            return 'models/random_forest'
        elif self.classifier_type == ClassifierType.SVM:
            return 'models/svm'
        else:
            raise ValueError(f"Invalid classifier type: {self.classifier_type}")

    def save_model(self, model_filename, vectoriser_filename):
        directory_path = self.get_directory_path()
        model_filepath = f"{directory_path}/{model_filename}"
        vectoriser_filepath = f"{directory_path}/{vectoriser_filename}"

        self.logger.info(f'Saving model to {model_filepath}')
        self.classifier.save_model(model_filepath, vectoriser_filepath)
        self.logger.info('Model saved.')
