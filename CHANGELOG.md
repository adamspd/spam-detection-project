# Changelog

All notable changes to this project are documented in this file.

## [2.3.0] - 2026-08-08

### Fixed

- **XGBoost votes in `VotingSpamDetector` were silently always "ham", regardless of the message.**
  This was a regression from the `_predict_label` helper introduced in 2.2.0, caused by two
  compounding bugs:
  1. `_predict_label` called the raw `XGBClassifier.predict()` directly. Unlike the other 4
     classifiers (which are fit on the string labels `'ham'`/`'spam'` directly), XGBoost is fit on
     labels encoded to `0`/`1` via a `LabelEncoder`, so its raw prediction was never decoded back
     to a string, and `prediction == 'spam'` was always `False`.
  2. `_predict_label` densified the vectorised message with `.toarray()` before predicting.
     XGBoost's default handling of "missing" values differs between a sparse matrix (where an
     implicit zero can be treated as missing) and a dense array (where every value, including
     `0.0`, is a real one), which skewed the underlying model's own predictions once decoding was
     fixed. The message is now kept sparse, matching how all 5 classifiers are trained.
  - Practical impact: XGBoost's ~20% share of the weighted vote never actually contributed a
    "spam" vote in any deployed 2.2.0 installs. `VotingSpamDetector` still worked, but leaned
    more heavily on the other 4 classifiers than the documented weights implied.
- `spam_detector_ai/tests/py_test.py` previously excluded `XGB` from its parametrised accuracy
  tests, which is why this shipped undetected. `ClassifierType.XGB` has been added to the test
  matrix (both the live-training and committed-model accuracy tests) so this class of regression
  is caught in CI going forward.

### Changed

- All 5 models were retrained on the current dependency stack and the current `spam.csv` dataset;
  accuracy numbers in the README and `ModelAccuracy` (`prediction/performance.py`) — and therefore
  the normalised voting weights — were refreshed to match.
- Updated pinned dev/test dependencies in `requirements.txt` to their latest available releases:
  `numpy` (2.5.1), `xgboost` (3.4.0), `nltk` (3.10.2), `joblib` (1.5.3).
- `setup.cfg`: `python_requires` raised from `>=3.8` to `>=3.11` and the Python trove classifiers
  updated to 3.11-3.14. `scikit-learn>=1.7.0` (the floor `install_requires` already declared) only
  ships wheels for Python >=3.11, so anything below that was already unable to install the
  package's actual dependencies; the metadata now reflects that. The `pandas` upper bound was
  raised from `<3.0.0` to `<4.0.0` to allow pandas 3.x.
- CI (`.github/workflows/tests.yml`) now runs the test suite on a matrix of Python 3.11, 3.12,
  3.13, and 3.14 instead of only 3.13.

## [2.2.0] - 2026-08-08

### Added

- `VotingSpamDetector.score(message_) -> SpamScore`: returns a structured, JSON-serialisable
  breakdown of a classification instead of just a boolean — `is_spam`, `score` (weighted vote
  fraction, 0.0-1.0), `threshold`, `score_type` (`"weighted_vote"`), and a `votes` list with one
  entry per classifier (`classifier`, `vote`, `weight`). `score` is a weighted vote fraction, **not**
  a calibrated probability — see the README for why.

### Changed

- `VotingSpamDetector.is_spam` is now a thin wrapper over `score(message_).is_spam`. Its signature,
  return type, and decision behaviour are unchanged; it now runs the 5 underlying classifiers once
  per call instead of running them a second time whenever `score()` is also needed.
- `VotingSpamDetector.is_spam` no longer `print()`s a vote breakdown to stdout on every call. The
  same information is now emitted via `logging` at `DEBUG` level (logger
  `spam_detector_ai.prediction.predict`), lazily formatted so there's no cost when the level is off.
- The decision threshold used by the returned boolean and the one used in the (now removed) log
  line are computed from the same value, instead of one being derived from the normalised weights
  and the other hardcoding `0.5`.
- Preprocessing (`Preprocessor.preprocess_text`) now runs once per `VotingSpamDetector.score()` /
  `is_spam()` call instead of once per underlying classifier (5x fewer preprocessing passes).
  `SpamDetector.is_spam` gained an optional `preprocessed=False` keyword to support this; existing
  callers passing raw text are unaffected.
- `SpamDetector.is_spam` / `VotingSpamDetector.is_spam` now return a native Python `bool` rather
  than a `numpy.bool_`, so results are safe to pass straight into `json.dumps` or Django's
  `JsonResponse`.

### Internal

- `SpamDetector.is_spam` and `SpamDetector.test_is_spam` now share a `_predict_label` helper
  instead of duplicating the vectorise-and-predict steps.

## [2.1.19] and earlier

See [GitHub Releases](https://github.com/adamspd/spam-detection-project/releases).
