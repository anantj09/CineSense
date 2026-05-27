import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

from src.utils.logger import logger
from src.exception import CustomException
from src.utils.common import read_yaml
model_config = read_yaml("configs/model_config.yaml")

import sys

class DataPreprocessing:
    def __init__(self):
        self.max_sequence_length = model_config["max_sequence_length"]

    def pad_dataset(self, x_train, x_test):
        try:
            logger.info("Padding sequences started")

            x_train = pad_sequences(
                x_train,
                maxlen=self.max_sequence_length
            )
            x_test = pad_sequences(
                x_test,
                maxlen=self.max_sequence_length
            )

            logger.info("Padding completed successfully")

            return x_train, x_test

        except Exception as e:
            raise CustomException(e, sys)

    def get_word_index(self):
        try:
            logger.info("Loading word index")

            word_index = imdb.get_word_index()

            logger.info("Word index loaded successfully")

            return word_index

        except Exception as e:
            raise CustomException(e, sys)
        
    def decode_reviews(self, encoded_reviews):
        try:
            logger.info("Decoding IMDb reviews")

            word_index = imdb.get_word_index()
            reverse_word_index = {value: key for key, value in word_index.items()}
            decoded_reviews = []
            for review in encoded_reviews:
                decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
                decoded_reviews.append(decoded_review)

            logger.info("Reviews decoded successfully")

            return decoded_reviews

        except Exception as e:
            raise CustomException(e, sys)