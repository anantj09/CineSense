import os
import sys
import json

import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException

training_config = read_yaml("configs/training_config.yaml")
model_config = read_yaml("configs/model_config.yaml")

class HyperparameterTuner:
    def __init__(self, x_train, y_train, x_test, y_test, model_type="LSTM"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type
        self.max_features = model_config["max_features"]
        self.max_sequence_length = model_config["max_sequence_length"]

        os.makedirs("experiments", exist_ok=True)

    def build_model(self, trial):
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        units = trial.suggest_categorical("units", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = Sequential()
        model.add(
            Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=self.max_sequence_length,
            )
        )

        # Dynamic Architecture Selection
        if self.model_type == "LSTM":
            model.add(LSTM(units, dropout=dropout))
        elif self.model_type == "GRU":
            model.add(GRU(units, dropout=dropout))
        elif self.model_type == "BiLSTM":
            model.add(Bidirectional(LSTM(units, dropout=dropout)))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        model.add(Dense(1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss= training_config["loss"],
            metrics= training_config["metrics"],
        )

        return model

    def objective(self, trial):
        try:
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            model = self.build_model(trial)
            history = model.fit(
                self.x_train,
                self.y_train,
                epochs = training_config["hyper_epochs"],
                batch_size = training_config["hyper_batch_size"],
                validation_split = training_config["validation_split"],
                verbose=0,
            )
            loss, accuracy = model.evaluate(
                self.x_test, self.y_test, verbose=0
            )

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def run_optimization(self, n_trials=5):
        try:
            logger.info(f"Starting tuning for {self.model_type}")

            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=n_trials)

            logger.info(f"Tuning completed for {self.model_type}")

            logger.info(f"Best Params: {study.best_params}")

            logger.info(f"Best Accuracy: {study.best_value}")

            save_path = os.path.join("experiments", f"best_{self.model_type.lower()}_params.json")
            with open(save_path, "w") as f:
                json.dump(study.best_params, f, indent=4)

            return study

        except Exception as e:
            raise CustomException(e, sys)
