import os
import sys
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException
model_config = read_yaml("configs/model_config.yaml")
paths_config = read_yaml("configs/paths_config.yaml")

class TokenizerBuilder:
    def __init__(self):
        self.max_features = model_config["max_features"]
        self.max_sequence_length = model_config["max_sequence_length"]
        self.tokenizer_dir = paths_config["tokenizer_dir"]
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")

    def fit_tokenizer(self, texts):
        try:
            logger.info("Fitting tokenizer")

            self.tokenizer.fit_on_texts(texts)

            logger.info("Tokenizer fitted successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def texts_to_padded_sequences(self, texts):
        try:
            logger.info("Converting texts to sequences")

            sequences = self.tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

            logger.info("Text sequences padded successfully")

            return padded_sequences

        except Exception as e:
            raise CustomException(e, sys)

    def save_tokenizer(self):
        try:
            tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.pkl")
            joblib.dump(self.tokenizer, tokenizer_path)

            logger.info(f"Tokenizer saved at {tokenizer_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def load_tokenizer(self):
        try:
            tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.pkl")
            tokenizer = joblib.load(tokenizer_path)

            logger.info("Tokenizer loaded successfully")

            return tokenizer

        except Exception as e:
            raise CustomException(e, sys)