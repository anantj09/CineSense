import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout
)

from src.utils.logger import logger
from src.utils.common import read_yaml
training_config = read_yaml("configs/training_config.yaml")
model_config = read_yaml("configs/model_config.yaml")

class SimpleRNNModel:
    def __init__(self):
        self.max_features = model_config["max_features"]
        self.embedding_dim = model_config["embedding_dim"]
        self.max_sequence_length = model_config["max_sequence_length"]
        self.units = model_config["simple_rnn"]["units"]
        self.dropout = model_config["simple_rnn"]["dropout"]
        self.recurrent_dropout = model_config["simple_rnn"]["recurrent_dropout"]

    def build_model(self):
        logger.info("Building Simple RNN model")

        model = Sequential([
            Embedding(
                input_dim=self.max_features,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            SimpleRNN(
                self.units,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                activation='tanh'
            ),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer= training_config["optimizer"],
            loss= training_config["loss"],
            metrics= training_config["metrics"]
        )

        logger.info("Simple RNN model built successfully")

        return model