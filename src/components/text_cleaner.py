import re
import string
import sys

from bs4 import BeautifulSoup
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils.logger import logger
from src.exception import CustomException

nltk.download('stopwords')
nltk.download('wordnet')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text, remove_stopwords=False, apply_lemmatization=True):
        try:
            logger.info("Text cleaning started")
            
            text = text.lower()

            # Remove HTML tags
            text = BeautifulSoup(text, "html.parser").get_text()
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', "", text)
            
            # Remove numbers
            text = re.sub(r'\d+', '',text)

            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            
            # Remove stopwords
            if remove_stopwords:
                words = [word for word in words if word not in self.stop_words]
            
            # Lemmatization
            if apply_lemmatization:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            cleaned_text = ' '.join(words)

            logger.info("Text cleaning completed")

            return cleaned_text

        except Exception as e:
            raise CustomException(e, sys)