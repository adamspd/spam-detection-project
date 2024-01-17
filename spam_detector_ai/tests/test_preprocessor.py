import unittest
from unittest.mock import patch

from spam_detector_ai.loading_and_processing.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = Preprocessor()

    @patch('nltk.corpus.stopwords.words')
    def test_remove_special_characters(self, mock_stopwords):
        mock_stopwords.return_value = ['this', 'is', 'a', 'stopword']
        text = "Hello!!!, he said ---and went."
        processed = self.preprocessor.preprocess_text(text)
        self.assertNotIn("!", processed)
        self.assertNotIn("-", processed)

    @patch('nltk.corpus.stopwords.words')
    def test_to_lowercase(self, mock_stopwords):
        mock_stopwords.return_value = ['this', 'is', 'a', 'stopword']
        text = "This SHOULD be lowerCASE."
        processed = self.preprocessor.preprocess_text(text)
        self.assertEqual(processed, processed.lower())

    @patch('nltk.corpus.stopwords.words')
    def test_remove_stopwords(self, mock_stopwords):
        mock_stopwords.return_value = ['this', 'is', 'a', 'stopword']
        text = "This is a test."
        processed = self.preprocessor.preprocess_text(text)
        self.assertNotIn("is", processed.split())

    @patch('nltk.corpus.stopwords.words')
    def test_lemmatization(self, mock_stopwords):
        mock_stopwords.return_value = ['and', 'the']
        text = "running runs"
        processed = self.preprocessor.preprocess_text(text)
        self.assertIn("run", processed)
        self.assertIn("running", processed)
        self.assertNotIn("runs", processed)

    @patch('nltk.corpus.stopwords.words')
    def test_empty_string(self, mock_stopwords):
        mock_stopwords.return_value = ['stopword']
        text = ""
        processed = self.preprocessor.preprocess_text(text)
        self.assertEqual(processed, "")


if __name__ == '__main__':
    unittest.main()
