from spam_detector_ai.classifiers import ClassifierType, NaiveBayesClassifier, RandomForestSpamClassifier, SVMClassifier

CLASSIFIER_MAP = {
    ClassifierType.NAIVE_BAYES.value: NaiveBayesClassifier(),
    ClassifierType.RANDOM_FOREST.value: RandomForestSpamClassifier(),
    ClassifierType.SVM.value: SVMClassifier()
}
