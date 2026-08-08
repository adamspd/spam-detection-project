# Spam-Detector-AI

![Tests](https://github.com/adamspd/spam-detection-project/actions/workflows/tests.yml/badge.svg)
[![Current Release Version](https://img.shields.io/github/release/adamspd/spam-detection-project.svg?style=flat-square&logo=github)](https://github.com/adamspd/spam-detection-project/releases)
[![pypi Version](https://img.shields.io/pypi/v/spam-detector-ai.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spam-detector-ai/)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/spam-detector-ai?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/spam-detector-ai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/adamspd/spam-detection-project)](https://github.com/adamspd/spam-detection-project/commit/main)
[![GitHub issues](https://img.shields.io/github/issues/adamspd/spam-detection-project)](https://github.com/adamspd/spam-detection-project/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/adamspd/spam-detection-project)](https://github.com/adamspd/spam-detection-project/pulls)
[![GitHub contributors](https://img.shields.io/github/contributors/adamspd/spam-detection-project)](https://github.com/adamspd/spam-detection-project/graphs/contributors)

Spam-Detector-AI is a Python package for detecting and filtering spam messages using Machine Learning models. The
package integrates with Django or any other project that uses python and offers different types of classifiers: Naive
Bayes, Random Forest, and Support Vector Machine (SVM). Since version 2.1.0, two new classifiers have been added:
Logistic Regression and XGBClassifier.

⚠️ _**Warning**: No significant breaking changes were added to the version 2.x.x in terms of usage. On the other hand,
the fine-tuning of the models has been moved to a separate module (`tuning`) and the tests have been moved to a
separate module (`tests`)._
⚠️

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    - [Training the Models](#training-the-models)
    - [Tests](#tests)
    - [Making Predictions](#making-predictions)
        - [Getting a confidence score](#getting-a-confidence-score)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Notes](#notes)

## Installation

You can install the spam detection package via pip:

```sh
pip install spam-detector-ai
```

Make sure you have the following dependencies installed:

- scikit-learn
- nltk
- pandas
- numpy
- joblib
- xgboost

Additionally, you'll need to download the NLTK data and to do so, use the python interpreter to run the following
commands:

```python
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
```

## Usage

### Training the Models

Before using the classifiers, you must train the models. Training data is loaded from a CSV file. You can find the
training data in the `data` directory in the GitHub's page of the project. The CSV file must have 3 columns: `label`,
`text` and `label_num`. The `text` column contains the content of the message to analyze and the `label` column
contains the labels `ham` or `spam` and `label_num` contains the number `0` (not spam) or `1`(spam).

The more data you have, the better the models will perform.

To train the models, run the following command:

```sh
python3 spam_detector_ai/trainer.py
```

⚠️ **Warning**: A module not found error may occur ⚠️

If this happens, use an IDE to run the `trainer.py`file until a fix is implemented.

This will train all the models and save them as `.joblib` files in the models directory. For now, there is 3 models:

- `naive_bayes_model.joblib`
- `random_forest_model.joblib`
- `svm_model.joblib`
- `logistic_regression_model.joblib`
- `xgb_model.joblib`

### Tests

The test results are shown below:

#### _Model: NAIVE_BAYES_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|-----------------------|----------------------|
| **Actual: Ham**  | 2044 (True Negative)  | 101 (False Positive) |
| **Actual: Spam** | 38 (False Negative)   | 1106 (True Positive) |

- True Negative (TN): 2044 messages were correctly identified as ham (non-spam).
- False Positive (FP): 101 ham messages were incorrectly identified as spam.
- False Negative (FN): 38 spam messages were incorrectly identified as ham.
- True Positive (TP): 1106 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.9817    | 0.9529 | 0.9671   | 2145    |
| Spam         | 0.9163    | 0.9668 | 0.9409   | 1144    |
| **Accuracy** |           |        | **0.9577** | 3289  |
| Macro Avg    | 0.9490    | 0.9598 | 0.9540   | 3289    |
| Weighted Avg | 0.9590    | 0.9577 | 0.9580   | 3289    |

##### Accuracy: 0.9577379142596534

<br>

#### _Model: RANDOM_FOREST_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|-----------------------|----------------------|
| **Actual: Ham**  | 2094 (True Negative)  | 51 (False Positive)  |
| **Actual: Spam** | 33 (False Negative)   | 1111 (True Positive) |

- True Negative (TN): 2094 messages were correctly identified as ham (non-spam).
- False Positive (FP): 51 ham messages were incorrectly identified as spam.
- False Negative (FN): 33 spam messages were incorrectly identified as ham.
- True Positive (TP): 1111 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.9845    | 0.9762 | 0.9803   | 2145    |
| Spam         | 0.9561    | 0.9712 | 0.9636   | 1144    |
| **Accuracy** |           |        | **0.9745** | 3289  |
| Macro Avg    | 0.9703    | 0.9737 | 0.9720   | 3289    |
| Weighted Avg | 0.9746    | 0.9745 | 0.9745   | 3289    |

##### Accuracy: 0.9744603222864092

<br>

#### _Model: SVM_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|-----------------------|----------------------|
| **Actual: Ham**  | 2093 (True Negative)  | 52 (False Positive)  |
| **Actual: Spam** | 31 (False Negative)   | 1113 (True Positive) |

- True Negative (TN): 2093 messages were correctly identified as ham (non-spam).
- False Positive (FP): 52 ham messages were incorrectly identified as spam.
- False Negative (FN): 31 spam messages were incorrectly identified as ham.
- True Positive (TP): 1113 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.9854    | 0.9758 | 0.9806   | 2145    |
| Spam         | 0.9554    | 0.9729 | 0.9641   | 1144    |
| **Accuracy** |           |        | **0.9748** | 3289  |
| Macro Avg    | 0.9704    | 0.9743 | 0.9723   | 3289    |
| Weighted Avg | 0.9750    | 0.9748 | 0.9748   | 3289    |

##### Accuracy: 0.9747643660687139

<br>

#### _Model: LOGISTIC_REGRESSION_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|-----------------------|----------------------|
| **Actual: Ham**  | 2060 (True Negative)  | 85 (False Positive)  |
| **Actual: Spam** | 53 (False Negative)   | 1091 (True Positive) |

- True Negative (TN): 2060 messages were correctly identified as ham (non-spam).
- False Positive (FP): 85 ham messages were incorrectly identified as spam.
- False Negative (FN): 53 spam messages were incorrectly identified as ham.
- True Positive (TP): 1091 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.9749    | 0.9604 | 0.9676   | 2145    |
| Spam         | 0.9277    | 0.9537 | 0.9405   | 1144    |
| **Accuracy** |           |        | **0.9580** | 3289  |
| Macro Avg    | 0.9513    | 0.9570 | 0.9541   | 3289    |
| Weighted Avg | 0.9585    | 0.9580 | 0.9582   | 3289    |

##### Accuracy: 0.958041958041958

<br>

#### _Model: XGB_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|-----------------------|----------------------|
| **Actual: Ham**  | 2050 (True Negative)  | 95 (False Positive)  |
| **Actual: Spam** | 26 (False Negative)   | 1118 (True Positive) |

- True Negative (TN): 2050 messages were correctly identified as ham (non-spam).
- False Positive (FP): 95 ham messages were incorrectly identified as spam.
- False Negative (FN): 26 spam messages were incorrectly identified as ham.
- True Positive (TP): 1118 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.9875    | 0.9557 | 0.9713   | 2145    |
| Spam         | 0.9217    | 0.9773 | 0.9487   | 1144    |
| **Accuracy** |           |        | **0.9632** | 3289  |
| Macro Avg    | 0.9546    | 0.9665 | 0.9600   | 3289    |
| Weighted Avg | 0.9646    | 0.9632 | 0.9634   | 3289    |

##### Accuracy: 0.9632107023411371

All five models are now evaluated on the same 3289-row held-out split (`test_size=0.2, random_state=0`) of the
current training data. The SVM and Random Forest models perform best, with Logistic Regression and Naive Bayes
sitting a bit lower, and XGBoost in between.
Given that no single model achieved perfect accuracy, I have decided to implement a voting classifier.
This classifier will combine the predictions of the five models (Naive Bayes, Random Forest, SVM,
Logistic Regression, and XGB) using a majority vote system to make the final prediction.
This approach aims to leverage the strengths of each model to improve overall prediction accuracy.

##### Weighted Voting System

To enhance the decision-making process, I've refined our approach to a weighted voting system. This new system assigns
different weights to each model's vote based on their respective accuracies. The weights are proportional to the
accuracy of each model relative to the sum of the accuracies of all models. The models with higher accuracy have a
greater influence on the final decision.

The models and their respective proportional weights are as follows:

- Naive Bayes: Weight = 0.1983
- Random Forest: Weight = 0.2018
- SVM (Support Vector Machine): Weight = 0.2019
- Logistic Regression: Weight = 0.1984
- XGBoost (XGB): Weight = 0.1996

These weights were calculated based on the accuracy of each model as a proportion of the total accuracy of all models.
The final decision whether a message is spam or not is determined by the weighted spam score. Each model casts a vote
(spam or not spam), and this vote is multiplied by the model's weight. The weighted spam scores from all models are then
summed up. If this total weighted spam score exceeds 50% of the total possible weight, the message is classified as
spam. Otherwise, it's classified as not spam (ham).

This approach ensures that the more accurate models have a larger say in the final decision, thereby increasing the
reliability of spam detection. It combines the strengths of each model, compensating for individual weaknesses and
provides a more nuanced classification.

##### System Output

`VotingSpamDetector` logs the vote (spam or ham) from each model, along with its weight, the total weighted spam
score, and the final classification decision, at `DEBUG` level via the standard `logging` module (logger name
`spam_detector_ai.prediction.predict`). Nothing is printed to stdout as a side effect of classification — enable
`DEBUG` logging on that logger in your own application if you want to see this detail:

```python
import logging

logging.getLogger("spam_detector_ai.prediction.predict").setLevel(logging.DEBUG)
```

For a structured, programmatic breakdown of a single classification (rather than a log line), use `score()` —
see [Getting a confidence score](#getting-a-confidence-score) below.

If you have trained the models on new data, you can test them by running the following command:

```sh
python tests/test.py
```

⚠️ **Warning**: A module not found error may occur ⚠️

If this happens, use an IDE to run the `test.py`file until a fix is implemented.

### Making Predictions

To use the spam detector in your Django project:

1. Import the `VotingSpamDetector` from the `prediction` module.
2. Create an instance of the detector.
3. Use the `is_spam` method to check if a message is spam.

```python
from spam_detector_ai.prediction.predict import VotingSpamDetector

# Create the spam detector
spam_detector = VotingSpamDetector()

# Check if a message is spam
message = "Enter the message here"
is_spam = spam_detector.is_spam(message)
print(f"Is spam: {is_spam}")
```

### Getting a confidence score

If a plain `True`/`False` isn't enough — for example, you want to auto-file messages above some
confidence level and queue the rest for manual review — use `score()` instead of `is_spam()`. It runs the
same 5 classifiers once and returns a structured breakdown instead of throwing the number away:

```python
from spam_detector_ai.prediction.predict import VotingSpamDetector

spam_detector = VotingSpamDetector()

result = spam_detector.score("Enter the message here")
print(result.is_spam)     # bool   -- identical to spam_detector.is_spam(message)
print(result.score)       # float  -- weighted vote fraction, 0.0-1.0
print(result.threshold)   # float  -- the value `score` is compared against (0.5)
print(result.score_type)  # str    -- "weighted_vote" (see warning below)
print(result.votes)       # list   -- one entry per classifier: {"classifier", "vote", "weight"}

# JSON-serialisable, e.g. for an API response:
import json
json.dumps(result.as_dict())
```

`is_spam(message)` is unchanged — it now runs as a thin wrapper over `score(message).is_spam`, so
the two can never disagree, and calling both no longer runs the 5 classifiers twice.

⚠️ **`score` is not a probability** ⚠️

`score` is a **weighted vote fraction**, not a calibrated likelihood. Each of the 5 classifiers casts a
binary spam/ham vote. Each vote is multiplied by that classifier's accuracy-based weight
(`ModelAccuracy.X / total_accuracy`) — these weights are normalised to sum to `1.0`, which is why
`threshold` is `0.5` — and the weighted votes are summed. Because there are only 5 binary votes, there
are only 2⁵ = 32 possible combinations, so `score` can only ever land on one of at most 32
unevenly-spaced values. **A score of `0.82` does not mean "82% likely to be spam"** — it means "the
classifiers that voted spam together hold 82% of the total accuracy weight". Treat it as a ranking
signal for a threshold you tune empirically, not as a statistical probability.

`score_type` is included so that a future, genuinely probabilistic score (e.g. from `predict_proba`)
can be shipped on the same `0.0`-`1.0` scale under the same field name without silently changing what
an already-tuned threshold means. If you build automation around `score`, check `score_type ==
"weighted_vote"` before trusting a threshold you calibrated against it — a different value means the
number means something different now.

## Project Structure

- `classifiers/`: Contains the different classifiers (Naive Bayes, Random Forest, SVM, XGB & Logistic Regression).
- `data/`: Contains the sample dataset for training the classifiers.
- `loading_and_processing/`: Contains utility functions for loading and preprocessing data.
- `models/`: Contains the trained models and their vectorizers.
- `prediction/`: Contains the main spam detector class.
- `tests/`: Contains scripts for testing
- `tuning/`: Contains scripts for tuning the classifiers.
- `training/`: Contains scripts for training the classifiers.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.

## License

This project is licensed under the [MIT License](LICENSE).

## Notes

⚠️ **Warning**: In a future version, the data/spam.csv file may be removed to lighten the package. ⚠️

The project contains 5 pre-trained models that can be used directly if you want to skip the training step.
If you don't want to use the package, you can use the API that I have deployed
[here](https://spam-detection-api.adamspierredavid.com/).

## API Versions
⚠️ **Warning**: V1 is deprecated and V2 is highly recommended instead. ⚠️

There are two versions of the API:

- **v1:** The original version of the API that requires only the message to determine if it is a spam.
- **v2:** The updated version of the API that requires a JSON object with additional context about the message.

### Why Create a Second Version of the API?

The initial version of the API was too simplistic and did not provide enough context about the message to make an
accurate prediction. The new version addresses this limitation by requiring more comprehensive information to improve
prediction accuracy.

### Required Information for Version 2

To use version 2 of the API, the following information must be provided:

- `text`: The content of the message to analyze.
- `subject`: The subject of the message (optional).
- `email`: The sender's email address.
- `name`: The sender's name (optional).
- `user_agent`: The sender's user agent string (important but, optional).
- `ip`: The sender's IP address.

### Additional Checks and Enhancements

In version 2, several additional checks will be performed to enhance the accuracy of spam detection:

- **Email Validation:**
    - The email address format is validated.
    - The existence of the domain is verified.
    - MX, DMARC, and SPF records are checked.
    - The email is checked against a list of disposable email addresses.

If the models detect that the message is spam and the email is fake or disposable, the likelihood of the message being
spam increases. Similarly, if the IP address is known to be associated with spam activities, the message is more likely
to be classified as spam.

- **IP Address Check:**
    - The IP address is checked against known spam IP addresses.

- **User Agent Validation:**
    - The user agent string is validated.
    - If the user agent appears to be from a bot, the message is more likely to be spam.

By incorporating these additional checks, version 2 of the API provides a more robust and accurate spam detection
service.

## Example Usage of the API

### Version 2

<details>
<summary>Python Example</summary>

```python
import requests


def get_client_ip(request):
    ip = request.META.get('HTTP_X_FORWARDED_FOR')
    if ip:
        ip = ip.split(',')[-1].strip()
    else:
        ip = request.META.get('HTTP_X_REAL_IP') or request.META.get('REMOTE_ADDR')
    return ip


def check_website_contact_form_v2(request):
    message = request.POST.get('message')
    subject = request.POST.get('subject')
    email = request.POST.get('email')
    name = request.POST.get('name', '')  # Optional
    user_agent = request.META.get('HTTP_USER_AGENT', '')  # Optional
    ip = get_client_ip(request)  # Get the IP address of the sender

    # Prepare the JSON payload with the required information
    payload = {
        'text': message,
        'subject': subject,
        'email': email,
        'name': name,
        'user_agent': user_agent,
        'ip': ip
    }

    # Call the spam detection API
    response = requests.post(
            "https://spam-detection-api.adamspierredavid.com/v2/check-spam/",
            json=payload  # Use json parameter to send the payload
    )

    is_spam = False

    # Check if the API request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_response = response.json()
        is_spam = json_response.get('is_spam')

    if is_spam:
        # Do something if the message is classified as spam.
        # Personally, I save them as a Quarantined object to look at later (knowing that my ML models are not perfect), 
        # and I do nothing with them in my code.
        pass
    else:
        # Do something if the message is not classified as spam
        # Send email, save it to a database as whatever, etc.
        pass
```
</details>

<details>
<summary>Javascript Example</summary>

```javascript
async function getClientIP(req) {
    let ip = req.headers['x-forwarded-for'];
    if (ip) {
        ip = ip.split(',').pop().trim();
    } else {
        ip = req.headers['x-real-ip'] || req.connection.remoteAddress;
    }
    return ip;
}

async function checkWebsiteContactFormV2(req, res) {
    const message = req.body.message;
    const subject = req.body.subject;
    const email = req.body.email;
    const name = req.body.name || '';  // Optional
    const user_agent = req.headers['user-agent'] || '';  // Optional
    const ip = await getClientIP(req);  // Get the IP address of the sender

    // Prepare the JSON payload with the required information
    const payload = {
        text: message,
        subject: subject,
        email: email,
        name: name,
        user_agent: user_agent,
        ip: ip
    };

    try {
        // Call the spam detection API
        const response = await fetch("https://spam-detection-api.adamspierredavid.com/v2/check-spam/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        let is_spam = false;

        // Check if the API request was successful
        if (response.ok) {
            const jsonResponse = await response.json();
            is_spam = jsonResponse.is_spam;
        }

        if (is_spam) {
            // Do something if the message is classified as spam.
            // Personally, I save them as a Quarantined object to look at later (knowing that my ML models are not perfect),
            // and I do nothing with them in my code.
        } else {
            // Do something if the message is not classified as spam
            // Send email, save it to a database as whatever, etc.
        }
    } catch (error) {
        console.error('Error checking spam:', error);
        // Handle error
    }
}
```

</details>



### Version 1 ___DEPRECATED___

<details>

<summary>Python Example</summary>

```python
import requests


def check_website_contact_form(request):
    message_ = request.POST.get('message')
    subject = request.POST.get('subject')
    # etc...

    # Concatenate subject and message
    message_with_subject = f'subject: {subject}. {message_}'

    # Call the spam detection API
    response = requests.post(
            "https://spam-detection-api.adamspierredavid.com/v1/check-spam/",
            json={'text': message_with_subject}  # Use json parameter instead of data
            # You can also do `json={'message': message_with_subject}
    )

    is_spam = False

    # Check if the API request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_response = response.json()
        is_spam = json_response.get('is_spam')

    if is_spam:
        # Do something
        pass
    else:
        # Do something else
        pass
```
</details>

<details>
<summary>Javascript Example</summary>

```javascript
async function checkWebsiteContactForm(req, res) {
    const message = req.body.message;
    const subject = req.body.subject;

    // Concatenate subject and message
    const messageWithSubject = `subject: ${subject}. ${message}`;

    // Prepare the JSON payload
    const payload = {
        text: messageWithSubject
    };

    try {
        // Call the spam detection API
        const response = await fetch("https://spam-detection-api.adamspierredavid.com/v1/check-spam/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        let is_spam = false;

        // Check if the API request was successful
        if (response.ok) {
            const jsonResponse = await response.json();
            is_spam = jsonResponse.is_spam;
        }

        if (is_spam) {
            // Do something
        } else {
            // Do something else
        }
    } catch (error) {
        console.error('Error checking spam:', error);
        // Handle error
    }
}
```
</details>
