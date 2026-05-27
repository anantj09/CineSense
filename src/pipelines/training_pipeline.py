import sys
import pandas as pd

from src.components.dataset_builder import DatasetBuilder
from src.components.trainer import ModelTrainer
from src.components.evaluator import ModelEvaluator

from src.models.simple_rnn_model import SimpleRNNModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.bilstm_model import BiLSTMModel

from src.utils.logger import logger
from src.utils.visualization import VisualizationManager
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.dataset_builder = DatasetBuilder()
        self.evaluator = ModelEvaluator()
        self.visualizer = VisualizationManager()

    def run_pipeline(self):
        try:
            logger.info("Training pipeline started")

            x_train, y_train, x_test, y_test = (self.dataset_builder.prepare_dataset())

            models = {"SimpleRNN": SimpleRNNModel(), "LSTM": LSTMModel(), "GRU": GRUModel(), "BiLSTM": BiLSTMModel()}

            results = []
            for model_name, model_builder in models.items():
                logger.info(f"Training {model_name}")

                model = model_builder.build_model()
                trainer = ModelTrainer(model=model, model_name=model_name)
                history = trainer.train(x_train, y_train)
                self.visualizer.plot_training_history(history, model_name)

                metrics = self.evaluator.evaluate_model(model, x_test, y_test)
                metrics["model"] = model_name
                results.append(metrics)

            results_df = pd.DataFrame(results)
            results_df.to_csv("experiments/model_comparison.csv", index=False)
            self.visualizer.plot_model_comparison(results_df)

            logger.info("Training pipeline completed")

            return results_df

        except Exception as e:
            raise CustomException(e, sys)