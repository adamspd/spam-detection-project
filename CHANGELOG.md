# Changelog

All notable changes to this project are documented in this file.

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
