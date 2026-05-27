import sys
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
)

from src.components.text_cleaner import TextCleaner
from src.utils.common import read_yaml
from src.utils.logger import logger
from src.exception import CustomException
model_config = read_yaml("configs/model_config.yaml")
paths_config = read_yaml("configs/paths_config.yaml")

class UnifiedPredictionPipeline:
    _loaded_models = {}
    _loaded_tokenizers = {}

    def __init__(self):
        self.max_sequence_length = model_config["max_sequence_length"]
        self.cleaner = TextCleaner()
        self.models_dir = paths_config["models_dir"]
        self.tokenizer_dir = paths_config["tokenizer_dir"]
        self.distilbert_model_dir = paths_config["distilbert_model_dir"]
        self.load_resources()

    def load_resources(self):
        try:
            logger.info("Loading cached resources...")

            if "rnn_tokenizer" not in (self._loaded_tokenizers):
                self._loaded_tokenizers["rnn_tokenizer"] = joblib.load(
                    f"{self.tokenizer_dir}/tokenizer.pkl"
                )

            if "distilbert_tokenizer" not in (self._loaded_tokenizers):
                self._loaded_tokenizers["distilbert_tokenizer"] = (
                    DistilBertTokenizerFast.from_pretrained(
                        self.distilbert_model_dir
                    )
                )

            model_names = ["GRU", "LSTM", "BiLSTM", "SimpleRNN"]
            for model_name in model_names:
                if model_name not in (self._loaded_models):
                    model_path = f"{self.models_dir}/{model_name}.keras"
                    self._loaded_models[model_name] = load_model(model_path)

            if "DistilBERT" not in (self._loaded_models):
                self._loaded_models["DistilBERT"] = (
                    TFDistilBertForSequenceClassification.from_pretrained(
                        self.distilbert_model_dir
                    )
                )

            logger.info("All resources loaded successfully")
        
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_rnn_text(self, text):
        cleaned_text = self.cleaner.clean_text(text)
        tokenizer = self._loaded_tokenizers["rnn_tokenizer"]
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(
            sequence, maxlen=self.max_sequence_length
        )

        return padded_sequence

    def preprocess_transformer_text(self, text):
        cleaned_text = self.cleaner.clean_text(text)
        tokenizer = self._loaded_tokenizers["distilbert_tokenizer"]
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="tf",
        )

        return encoding

    def predict(self, text, model_type="GRU"):
        try:
            logger.info(f"Running inference using {model_type}")

            if model_type in ["SimpleRNN", "LSTM", "GRU", "BiLSTM"]:
                model = self._loaded_models[model_type]
                processed_text = self.preprocess_rnn_text(text)
                prediction = model.predict(processed_text, verbose=0)[0][0]
                confidence = float(prediction)
                sentiment = "Positive" if confidence >= 0.5 else "Negative"

            elif model_type == "DistilBERT":
                model = self._loaded_models["DistilBERT"]
                processed_text = self.preprocess_transformer_text(text)
                try:
                    outputs = model(processed_text)
                except Exception as transformer_error:
                    logger.error(f"Transformer inference failed: {transformer_error}")
                    return {
                        "model": model_type,
                        "sentiment": "Inference Failed",
                        "confidence": 0.0,
                    }
                logits = outputs.logits.numpy()
                probabilities = tf.nn.softmax(logits, axis=1).numpy()
                prediction = np.argmax(probabilities, axis=1)[0]
                confidence = float(np.max(probabilities))
                sentiment = "Positive" if prediction == 1 else "Negative"

            else:
                raise ValueError(f"Unsupported model: {model_type}")

            result = {
                "model": model_type,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
            }

            logger.info(f"Inference Result: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)
