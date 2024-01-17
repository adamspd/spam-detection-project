from spam_detector_ai.classifiers import ClassifierType, NaiveBayesClassifier, RandomForestSpamClassifier, SVMClassifier

CLASSIFIER_MAP = {
    ClassifierType.NAIVE_BAYES: NaiveBayesClassifier,
    ClassifierType.RANDOM_FOREST: RandomForestSpamClassifier,
    ClassifierType.SVM: SVMClassifier
}
