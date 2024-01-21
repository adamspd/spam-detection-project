# performance.py

class ModelAccuracy:
    NAIVE_BAYES = 0.8679
    RANDOM_FOREST = 0.9750
    SVM = 0.9774
    LOGISTIC_REG = 0.9708
    XGB = 0.9711

    @classmethod
    def total_accuracy(cls):
        return sum([cls.NAIVE_BAYES, cls.RANDOM_FOREST, cls.SVM, cls.LOGISTIC_REG, cls.XGB])
