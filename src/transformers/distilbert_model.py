import sys

from transformers import TFDistilBertForSequenceClassification

from src.utils.logger import logger
from src.exception import CustomException

class DistilBERTModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name

    def load_model(self):
        try:
            logger.info("Loading DistilBERT model")

            model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

            logger.info("DistilBERT loaded successfully")

            return model

        except Exception as e:
            raise CustomException(e, sys)
