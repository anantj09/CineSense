import tensorflow as tf
from tensorflow.keras.datasets import imdb

from src.utils.logger import logger
from src.exception import CustomException
from src.utils.common import read_yaml
model_config = read_yaml("configs/model_config.yaml")

import sys

class DataIngestion:
    def __init__(self):
        self.max_features = model_config["max_features"]

    def load_imdb_dataset(self):
        try:
            logger.info("Loading IMDb dataset")
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)

            logger.info("IMDb dataset loaded successfully")

            return (
                x_train,
                y_train,
                x_test,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)