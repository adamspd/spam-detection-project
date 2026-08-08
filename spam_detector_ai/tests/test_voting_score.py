import json

import pytest

from spam_detector_ai.prediction.predict import SpamScore, VotingSpamDetector

MESSAGES = [
    "Hello!",
    "Hi, can we reschedule our meeting to next Tuesday at 3pm?",
    "Buy cheap viagra now, click here to win free money!!! Limited offer, act now.",
    "Thanks for sending over the report, I will review it tomorrow morning.",
]


@pytest.fixture(scope="module")
def detector():
    """Loading the 5 underlying models is slow, so build the detector once
    per module and reuse it across every test below."""
    return VotingSpamDetector()


class TestScoreAndIsSpamAgree:
    @pytest.mark.parametrize("message", MESSAGES)
    def test_boolean_matches(self, detector, message):
        assert detector.is_spam(message) == detector.score(message).is_spam

    @pytest.mark.parametrize("message", MESSAGES)
    def test_is_spam_returns_native_bool(self, detector, message):
        # A numpy.bool_ leaking out here would break JSON serialisation
        # downstream (e.g. Django's JsonResponse).
        assert type(detector.is_spam(message)) is bool


class TestScoreShape:
    @pytest.mark.parametrize("message", MESSAGES)
    def test_score_in_unit_interval(self, detector, message):
        result = detector.score(message)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.parametrize("message", MESSAGES)
    def test_result_is_json_serialisable(self, detector, message):
        result = detector.score(message)
        json.dumps(result.as_dict())

    @pytest.mark.parametrize("message", MESSAGES)
    def test_votes_have_one_entry_per_classifier(self, detector, message):
        result = detector.score(message)
        assert len(result.votes) == len(detector.detectors)
        classifier_names = {vote["classifier"] for vote in result.votes}
        expected_names = {name for name, _, _ in detector.detectors}
        assert classifier_names == expected_names
        for vote in result.votes:
            assert isinstance(vote["vote"], bool)
            assert isinstance(vote["weight"], float)

    def test_weights_sum_to_one(self, detector):
        total_weight = sum(weight for _, _, weight in detector.detectors)
        assert total_weight == pytest.approx(1.0)

    @pytest.mark.parametrize("message", MESSAGES)
    def test_score_type_is_weighted_vote(self, detector, message):
        assert detector.score(message).score_type == "weighted_vote"

    @pytest.mark.parametrize("message", MESSAGES)
    def test_threshold_is_half(self, detector, message):
        assert detector.score(message).threshold == pytest.approx(0.5)


class TestUnanimousVotes:
    """Rather than hunting for real-world text every one of the 5 classifiers
    happens to agree on, force each classifier's vote via monkeypatching so
    the scoring math itself is exercised deterministically."""

    def _force_votes(self, monkeypatch, detector, vote):
        for _, spam_detector, _ in detector.detectors:
            monkeypatch.setattr(spam_detector, "is_spam", lambda *args, **kwargs: vote)

    def test_unanimous_spam_scores_one(self, detector, monkeypatch):
        self._force_votes(monkeypatch, detector, True)
        result = detector.score("irrelevant, votes are forced")
        assert result.score == pytest.approx(1.0)
        assert result.is_spam is True

    def test_unanimous_ham_scores_zero(self, detector, monkeypatch):
        self._force_votes(monkeypatch, detector, False)
        result = detector.score("irrelevant, votes are forced")
        assert result.score == pytest.approx(0.0)
        assert result.is_spam is False


class TestNoStdoutOutput:
    @pytest.mark.parametrize("message", MESSAGES)
    def test_is_spam_writes_nothing_to_stdout(self, detector, message, capsys):
        detector.is_spam(message)
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.parametrize("message", MESSAGES)
    def test_score_writes_nothing_to_stdout(self, detector, message, capsys):
        detector.score(message)
        captured = capsys.readouterr()
        assert captured.out == ""


def test_spam_score_is_a_dataclass_instance(detector):
    assert isinstance(detector.score("Hello!"), SpamScore)
