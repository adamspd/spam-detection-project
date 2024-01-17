# spam_detector_ai/loading_and_processing/preprocessor.py

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))
        self.special_char_pattern = re.compile('[^a-zA-Z]')

    def preprocess_text(self, text):
        """
        Preprocess the text by removing special characters, converting to lowercase,
        removing stopwords, and lemmatizing.
        """
        # Remove all the special characters
        text = self.special_char_pattern.sub(' ', text)
        text = text.lower()
        text = text.split()
        # Remove stop words and lemmatise
        text = [self.lemmatiser.lemmatize(word) for word in text if word not in self.stopwords_set]
        text = ' '.join(text)
        return text

    def preprocess(self, data):
        """
        Apply text preprocessing to a DataFrame column 'text'.
        """
        if 'text' not in data.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        return data
