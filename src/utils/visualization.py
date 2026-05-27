import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.utils.logger import logger
from src.utils.common import read_yaml
from src.exception import CustomException
paths_config = read_yaml("configs/paths_config.yaml")

class VisualizationManager:
    def __init__(self):
        self.plots_dir = paths_config["plots_dir"]
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_model_comparison(self, results_df):
        try:
            logger.info("Creating model comparison plot")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=results_df, x="model", y="accuracy")
            plt.title("Model Accuracy Comparison")
            plt.ylabel("Accuracy")
            plt.xlabel("Model")
            save_path = os.path.join(self.plots_dir, "model_comparison.png")
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Model comparison plot saved at {save_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def plot_training_history(self, history, model_name):
        try:
            logger.info(f"Plotting training history for {model_name}")

            # Accuracy Plot
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            accuracy_path = os.path.join(self.plots_dir, f"{model_name}_accuracy.png")
            plt.savefig(accuracy_path)
            plt.close()

            # Loss Plot
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_path = os.path.join(self.plots_dir, f"{model_name}_loss.png")
            plt.savefig(loss_path)
            plt.close()

            logger.info(f"Training plots saved for {model_name}")

        except Exception as e:
            raise CustomException(e, sys)