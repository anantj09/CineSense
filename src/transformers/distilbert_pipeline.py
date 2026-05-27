import sys
import tensorflow as tf

from src.components.dataset_builder import DatasetBuilder
from src.components.preprocessing import DataPreprocessing
from src.components.text_cleaner import TextCleaner
from src.transformers.distilbert_dataset import DistilBERTDatasetBuilder
from src.transformers.distilbert_model import DistilBERTModel
from src.transformers.distilbert_trainer import DistilBERTTrainer
from src.transformers.distilbert_evaluator import DistilBERTEvaluator

from src.utils.logger import logger
from src.utils.common import read_yaml
from src.exception import CustomException
paths_config = read_yaml("configs/paths_config.yaml")

class DistilBERTPipeline:
    def __init__(self):
        self.dataset_builder = DatasetBuilder()
        self.preprocessing = DataPreprocessing()
        self.cleaner = TextCleaner()
        self.dataset_tokenizer = DistilBERTDatasetBuilder()
        self.distilbert_model_dir = paths_config["distilbert_model_dir"]

    def prepare_text_dataset(self):
        try:
            logger.info("Preparing text dataset for DistilBERT")

            x_train, y_train, x_test, y_test = (self.dataset_builder.data_ingestion.load_imdb_dataset())

            decoded_x_train = self.preprocessing.decode_reviews(x_train)
            decoded_x_test = self.preprocessing.decode_reviews(x_test)

            cleaned_x_train = [self.cleaner.clean_text(review) for review in decoded_x_train]
            cleaned_x_test = [self.cleaner.clean_text(review) for review in decoded_x_test]

            logger.info("Text dataset prepared successfully")

            return (cleaned_x_train, y_train, cleaned_x_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)

    def build_tf_dataset(self, encodings, labels, batch_size=8, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)

        return dataset

    def run_pipeline(self):
        try:
            logger.info("Starting DistilBERT pipeline")

            x_train, y_train, x_test, y_test = self.prepare_text_dataset()

            train_encodings = self.dataset_tokenizer.tokenize_dataset(x_train)
            test_encodings = self.dataset_tokenizer.tokenize_dataset(x_test)

            train_dataset = self.build_tf_dataset(train_encodings, y_train, shuffle=True)
            test_dataset = self.build_tf_dataset(test_encodings, y_test, shuffle=False)

            model_builder = DistilBERTModel()
            model = model_builder.load_model()
            trainer = DistilBERTTrainer(model=model)
            history = trainer.train(train_dataset, test_dataset)

            model.save_pretrained(self.distilbert_model_dir)
            self.dataset_tokenizer.tokenizer.save_pretrained(self.distilbert_model_dir)

            evaluator = DistilBERTEvaluator()
            metrics = evaluator.evaluate(
                model,
                test_dataset
            )
            print(metrics)

            logger.info("DistilBERT pipeline completed")

            return history

        except Exception as e:
            raise CustomException(e, sys)
