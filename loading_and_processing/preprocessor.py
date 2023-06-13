# preprocessor.py

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = re.sub('[^a-zA-Z]', ' ', text)  # Remove all the special characters
        text = text.lower()  # Convert to lowercase
        text = text.split()  # Split into words
        text = [self.lemmatiser.lemmatize(word) for word in text if
                not word in set(stopwords.words('english'))]  # Remove stop words and lemmatise
        text = ' '.join(text)
        return text

    def preprocess(self, data):
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        return data
