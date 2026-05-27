import sys
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException
model_config = read_yaml("configs/model_config.yaml")

class InferencePreprocessor:
    def __init__(self):
        self.max_sequence_length = model_config["max_sequence_length"]
        self.word_index = imdb.get_word_index()

    def encode_review(self,review):
        try:
            logger.info("Encoding user review")

            words = review.lower().split()
            encoded_review = [
                self.word_index.get(word, 2) + 3
                for word in words
            ]
            padded_review = pad_sequences([encoded_review],maxlen=self.max_sequence_length)

            logger.info("Review encoded successfully")

            return padded_review

        except Exception as e:
            raise CustomException(e, sys)