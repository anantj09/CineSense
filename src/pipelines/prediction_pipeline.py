"""
Legacy prediction pipeline.

Retained for backward compatibility.
Use unified_prediction_pipeline.py
for production inference.
"""


import sys

import numpy as np
from tensorflow.keras.models import load_model

from src.components.inference import (
    InferencePreprocessor
)

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException
paths_config = read_yaml("configs/paths_config.yaml")

class PredictionPipeline:
    def __init__(self):
        self.model_path = self.model_path = (f"{paths_config['models_dir']}/simple_rnn_test.keras")
        logger.info("Loading trained model")

        self.model = load_model(self.model_path)
        logger.info("Model loaded successfully")

        self.preprocessor = InferencePreprocessor()

    def predict_sentiment(self,review):
        try:
            processed_review = (
                self.preprocessor.encode_review(review)
            )
            prediction = self.model.predict(
                processed_review
            )
            confidence_score = float(
                prediction[0][0]
            )
            sentiment = (
                "Positive" if confidence_score >= 0.5
                else "Negative"
            )
            result = {
                "sentiment": sentiment,
                "confidence_score": round(confidence_score,4)
            }

            logger.info(f"Prediction result: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)