import sys
import tensorflow as tf

from transformers import create_optimizer
from tensorflow.keras import mixed_precision

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException

training_config = read_yaml("configs/training_config.yaml")

mixed_precision.set_global_policy('mixed_float16')

class DistilBERTTrainer:
    def __init__(self, model):
        self.model = model
        self.epochs = training_config["distilbert_epochs"]
        self.batch_size = training_config["distilbert_batch_size"]

    def train(self, train_dataset, val_dataset):
        try:
            logger.info("Starting DistilBERT training")

            batches_per_epoch = len(train_dataset)
            num_train_steps = batches_per_epoch * self.epochs

            optimizer, schedule = create_optimizer(
                init_lr=2e-5,
                num_train_steps=num_train_steps,
                num_warmup_steps=0,
            )
            self.model.compile(
                optimizer=optimizer,
                loss=self.model.hf_compute_loss,
                metrics= training_config["metrics"],
            )
            history = self.model.fit(
                train_dataset, validation_data=val_dataset, epochs=self.epochs
            )

            logger.info("DistilBERT training completed")

            return history

        except Exception as e:
            raise CustomException(e, sys)
