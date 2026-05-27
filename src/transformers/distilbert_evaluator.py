import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import logger
from src.exception import CustomException

class DistilBERTEvaluator:
    def evaluate(self, model, test_dataset):
        try:
            logger.info("Evaluating DistilBERT model")

            predictions = model.predict(test_dataset)
            logits = predictions.logits
            y_pred = np.argmax(logits, axis=1)

            y_true = []
            for _, labels in test_dataset:
                y_true.extend(labels.numpy())
            y_true = np.array(y_true)
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }
            logger.info(f"DistilBERT Metrics: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)
