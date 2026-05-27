import os
import sys

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras import mixed_precision

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException

training_config = read_yaml("configs/training_config.yaml")
paths_config = read_yaml("configs/paths_config.yaml")

class ModelTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.checkpoints_dir = paths_config["checkpoints_dir"]
        self.models_dir = paths_config["models_dir"]
        self.epochs = training_config["epochs"]
        self.batch_size = training_config["batch_size"]
        self.validation_split = training_config["validation_split"]

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_callbacks(self):
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"{self.model_name}.keras"
        )
        callbacks = [
            EarlyStopping(
                monitor= training_config["early_stopping"]["monitor"],
                patience= training_config["early_stopping"]["patience"],
                restore_best_weights= training_config["early_stopping"]["restore_best_weights"]
            ),
            ReduceLROnPlateau(
                monitor= training_config["reduce_lr"]["monitor"],
                factor= training_config["reduce_lr"]["factor"],
                patience= training_config["reduce_lr"]["patience"]
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]
        return callbacks

    def train(self,x_train,y_train):
        try:
            logger.info(f"Training started for {self.model_name}")

            mixed_precision.set_global_policy('mixed_float16')
            history = self.model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=self.get_callbacks()
            )

            model_save_path = os.path.join(
                self.models_dir,
                f"{self.model_name}.keras"
            )
            self.model.save(model_save_path)

            logger.info(f"Model saved at {model_save_path}")
            logger.info(f"Training completed for {self.model_name}")

            return history

        except Exception as e:
            raise CustomException(e, sys)