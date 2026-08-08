# spam_detector_ai/prediction/predict.py
"""
Author: Adams P. David
Contact: https://adamspierredavid.com/contact/
Date Written: 2023-06-12
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Union

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpamScore:
    """Structured result of ``VotingSpamDetector.score()``.

    ``score`` is a *weighted vote fraction*, not a calibrated probability.
    Each of the 5 underlying classifiers casts a binary spam/ham vote; each
    vote is multiplied by that classifier's normalised accuracy weight
    (``ModelAccuracy.X / total_accuracy``, and these weights sum to 1.0, so
    ``threshold`` is 0.5) and the results are summed. Because there are only
    2**5 = 32 possible combinations of 5 binary votes, ``score`` can only
    ever land on one of at most 32 unevenly-spaced values -- it is not a
    continuous, calibrated likelihood. A score of 0.82 does not mean "82%
    likely to be spam"; it means "the classifiers that voted spam together
    hold 82% of the total accuracy weight". Compare ``score`` against
    ``threshold`` (never hardcode 0.5) if you need to reproduce ``is_spam``,
    and use ``score`` directly only for ranking/thresholding, not as a
    probability.

    ``score_type`` is always ``"weighted_vote"`` today. It exists so a future
    probability-based score (e.g. via ``predict_proba``) can be introduced on
    the same 0.0-1.0 scale without silently changing what an already-tuned
    threshold means -- check ``score_type`` before reusing a stored
    threshold.
    """

    is_spam: bool
    score: float
    threshold: float
    score_type: str
    votes: List[Dict[str, Union[str, bool, float]]]

    def as_dict(self) -> dict:
        return asdict(self)


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

    def _predict_label(self, processed_message):
        """Vectorise already-preprocessed text and return the raw model label ('spam' or 'ham').

        Most classifiers are fit directly on string labels, so classifier.predict()
        already returns 'spam'/'ham'. XGBSpamClassifier is the exception: it fits on a
        LabelEncoder-encoded target, so its raw prediction is a numeric label that must
        be decoded back through that same encoder before it can be compared to 'spam'.

        The vectorised message is kept sparse (no ``.toarray()``): all 5 underlying
        classifiers are trained on the sparse TF-IDF matrix, and XGBoost in particular
        treats an explicit 0.0 in a *dense* array differently from an implicit zero in
        a *sparse* one (it can be read as a missing value), which silently skewed its
        predictions towards one class when this was densified.
        """
        vectorized_message = self.model.vectoriser.transform([processed_message])
        prediction = self.model.classifier.predict(vectorized_message)
        label_encoder = getattr(self.model, "label_encoder", None)
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform(prediction)
        return prediction[0]

    def is_spam(self, message_, preprocessed=False):
        """Return True if spam, False if not spam.

        By default ``message_`` is raw text and will be preprocessed here.
        Pass ``preprocessed=True`` if the caller (e.g. VotingSpamDetector)
        already ran the text through Preprocessor.preprocess_text, to avoid
        preprocessing the same message once per classifier.
        """
        processed_message = message_ if preprocessed else self.processor.preprocess_text(message_)
        # bool(...): the underlying model libraries return numpy scalar types
        # (e.g. numpy.bool_), which are not JSON-serialisable.
        return bool(self._predict_label(processed_message) == 'spam')

    def test_is_spam(self, message_):
        processed_message = self.processor.preprocess_text(message_)
        return self._predict_label(processed_message)


class VotingSpamDetector:
    """This class is used to detect whether a message is spam
    or not spam using majority voting of multiple spam detectors models."""

    def __init__(self):
        total_accuracy = ModelAccuracy.total_accuracy()
        self.detectors = [
            ("naive_bayes", SpamDetector(model_type=ClassifierType.NAIVE_BAYES),
             ModelAccuracy.NAIVE_BAYES / total_accuracy),
            ("random_forest", SpamDetector(model_type=ClassifierType.RANDOM_FOREST),
             ModelAccuracy.RANDOM_FOREST / total_accuracy),
            ("svm", SpamDetector(model_type=ClassifierType.SVM),
             ModelAccuracy.SVM / total_accuracy),
            ("logistic_regression", SpamDetector(model_type=ClassifierType.LOGISTIC_REGRESSION),
             ModelAccuracy.LOGISTIC_REG / total_accuracy),
            ("xgb", SpamDetector(model_type=ClassifierType.XGB),
             ModelAccuracy.XGB / total_accuracy),
        ]
        self.processor = Preprocessor()
        # Weights are normalised to sum to 1.0, so this is 0.5 -- computed rather than
        # hardcoded so the decision threshold can never drift from the weighting scheme.
        self.decision_threshold = sum(weight for _, _, weight in self.detectors) / 2

    def score(self, message_) -> SpamScore:
        """Classify ``message_`` and return the full weighted-vote breakdown.

        See ``SpamScore`` for what the returned ``score`` does and does not mean --
        in short, it is a weighted vote fraction, not a calibrated probability.
        """
        # Preprocess once and reuse across all 5 classifiers, instead of each
        # classifier independently preprocessing the same raw text.
        processed_message = self.processor.preprocess_text(message_)
        votes = [
            (name, detector.is_spam(processed_message, preprocessed=True), weight)
            for name, detector, weight in self.detectors
        ]
        weighted_spam_score = float(sum(vote * weight for _, vote, weight in votes))
        is_spam_result = bool(weighted_spam_score > self.decision_threshold)

        if logger.isEnabledFor(logging.DEBUG):
            vote_descriptions = [
                f"{name}: {'Spam' if vote else 'Ham'} (Weight: {weight:.4f})"
                for name, vote, weight in votes
            ]
            logger.debug(
                "Votes: %s, Weighted Spam Score: %.4f, Classified as: %s",
                vote_descriptions, weighted_spam_score, "Spam" if is_spam_result else "Ham",
            )

        return SpamScore(
            is_spam=is_spam_result,
            score=weighted_spam_score,
            threshold=self.decision_threshold,
            score_type="weighted_vote",
            votes=[{"classifier": name, "vote": vote, "weight": weight} for name, vote, weight in votes],
        )

    def is_spam(self, message_) -> bool:
        """Return True if spam, False otherwise. Thin wrapper over score() so the
        two can never disagree; see score() for the weighted score, threshold,
        and per-classifier vote breakdown."""
        return self.score(message_).is_spam


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
