# spam_detector_ai/loading_and_processing/preprocessor.py

import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))
        self.special_char_pattern = re.compile('[^a-zA-Z]')

    def update_dynamic_stopwords(self, data, threshold_ratio=0.8):
        """
        Updates the stopwords list based on the frequency of words in spam and ham messages.
        :param data: DataFrame with 'label' and 'basic_processed_text' columns
        :param threshold_ratio: The threshold for considering a word as non-discriminative
        :return: Set of dynamically identified stopwords
        """
        # Convert lists of words back to strings
        data['text_for_stopwords'] = data['basic_processed_text'].apply(lambda x: ' '.join(x))

        # Separate the data into spam and ham
        spam_data = data[data['label'] == 'spam']['text_for_stopwords']
        ham_data = data[data['label'] == 'ham']['text_for_stopwords']

        # Count word frequencies in each category
        spam_word_count = Counter(" ".join(spam_data).split())
        ham_word_count = Counter(" ".join(ham_data).split())

        # Identify common words based on a threshold ratio
        common_words = set()
        for word, freq in spam_word_count.items():
            ham_freq = ham_word_count.get(word, 0)
            total_freq = freq + ham_freq
            if min(freq, ham_freq) / total_freq > threshold_ratio:
                common_words.add(word)

        return common_words

    def update_stopwords(self, data):
        dynamic_stopwords = self.update_dynamic_stopwords(data)
        self.stopwords_set.update(dynamic_stopwords)

    def basic_preprocess_text(self, text):
        """
        Basic preprocessing: remove special characters and convert to lowercase.
        """
        text = self.special_char_pattern.sub(' ', text)
        return text.lower().split()

    def preprocess_text(self, text):
        """
        Preprocess the text by removing special characters, converting to lowercase,
        removing stopwords, and lemmatizing.
        """
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

        # Basic preprocessing
        data['basic_processed_text'] = data['text'].apply(self.basic_preprocess_text)

        # Update dynamic stopwords
        self.update_stopwords(data)

        # Full preprocessing
        data['processed_text'] = data['basic_processed_text'].apply(self.preprocess_text)
        return data
