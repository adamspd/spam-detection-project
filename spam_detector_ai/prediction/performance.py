# performance.py

class ModelAccuracy:
    NAIVE_BAYES = 0.9577
    RANDOM_FOREST = 0.9745
    SVM = 0.9748
    LOGISTIC_REG = 0.9580
    XGB = 0.9632

    @classmethod
    def total_accuracy(cls):
        return sum([cls.NAIVE_BAYES, cls.RANDOM_FOREST, cls.SVM, cls.LOGISTIC_REG, cls.XGB])
