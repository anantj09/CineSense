import sys

from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import DataPreprocessing
from src.components.text_cleaner import TextCleaner
from src.components.tokenizer_builder import TokenizerBuilder

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException

model_config = read_yaml("configs/model_config.yaml")

class DatasetBuilder:
    def __init__(self):
        self.max_features = model_config["max_features"]
        self.max_sequence_length = model_config["max_sequence_length"]
        self.data_ingestion = DataIngestion()
        self.preprocessing = DataPreprocessing()
        self.cleaner = TextCleaner()
        self.tokenizer_builder = TokenizerBuilder()

    def prepare_dataset(self):
        try:
            logger.info("Dataset preparation started")

            x_train, y_train, x_test, y_test = (self.data_ingestion.load_imdb_dataset())

            decoded_x_train = (self.preprocessing.decode_reviews(x_train))
            decoded_x_test = (self.preprocessing.decode_reviews(x_test))

            cleaned_x_train = [self.cleaner.clean_text(review) for review in decoded_x_train]
            cleaned_x_test = [self.cleaner.clean_text(review) for review in decoded_x_test]

            DEBUG_MODE = False
            if DEBUG_MODE:
                cleaned_x_train = cleaned_x_train[:5000]
                y_train = y_train[:5000]
                cleaned_x_test = cleaned_x_test[:2000]
                y_test = y_test[:2000]

            logger.info("Text cleaning completed")

            self.tokenizer_builder.fit_tokenizer(cleaned_x_train)

            x_train_padded = (self.tokenizer_builder.texts_to_padded_sequences(cleaned_x_train))
            x_test_padded = (self.tokenizer_builder.texts_to_padded_sequences(cleaned_x_test))

            self.tokenizer_builder.save_tokenizer()

            logger.info("Dataset preparation completed")

            return (
                x_train_padded,
                y_train,
                x_test_padded,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)