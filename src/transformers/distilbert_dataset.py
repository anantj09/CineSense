import sys

from transformers import DistilBertTokenizerFast

from src.utils.logger import logger
from src.exception import CustomException

class DistilBERTDatasetBuilder:
    def __init__(self, model_name="distilbert-base-uncased", max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)

    def tokenize_dataset(self, texts):
        try:
            logger.info("Tokenizing dataset for DistilBERT")

            encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length,)

            logger.info("DistilBERT tokenization completed")

            return encodings

        except Exception as e:
            raise CustomException(e, sys)
