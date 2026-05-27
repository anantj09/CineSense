import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.utils.logger import logger
from src.exception import CustomException

class ModelEvaluator:
    def evaluate_model(self,model,x_test,y_test):
        try:
            logger.info("Model evaluation started")

            predictions = model.predict(x_test)
            y_pred = (predictions > 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_test,y_pred),
                "precision": precision_score(y_test,y_pred,zero_division=0),
                "recall": recall_score(y_test,y_pred,zero_division=0),
                "f1_score": f1_score(y_test,y_pred,zero_division=0)
            }

            logger.info(f"Evaluation Metrics: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)