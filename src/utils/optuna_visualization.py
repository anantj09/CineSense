import os
import sys

import optuna.visualization as vis

from src.utils.logger import logger
from src.utils.common import read_yaml
from src.exception import CustomException
paths_config = read_yaml("configs/paths_config.yaml")

class OptunaVisualizer:
    def __init__(self):
        self.save_dir = paths_config["optuna_plots_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

    def save_optimization_history(self, study, model_name):
        try:
            logger.info(f"Saving optimization history for {model_name}")

            fig = vis.plot_optimization_history(study)
            save_path = os.path.join(self.save_dir, f"{model_name}_optimization_history.html")
            fig.write_html(save_path)

            logger.info(f"Saved optimization history at {save_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def save_param_importance(self, study, model_name):
        try:
            logger.info(f"Saving parameter importance for {model_name}")

            fig = vis.plot_param_importances(study)
            save_path = os.path.join(self.save_dir, f"{model_name}_param_importance.html")
            fig.write_html(save_path)

            logger.info(f"Saved parameter importance at {save_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def save_parallel_coordinate(self, study, model_name):
        try:
            logger.info(f"Saving parallel coordinate plot for {model_name}")

            fig = vis.plot_parallel_coordinate(study)
            save_path = os.path.join(self.save_dir, f"{model_name}_parallel_coordinate.html")
            fig.write_html(save_path)

            logger.info(f"Saved parallel coordinate plot at {save_path}")

        except Exception as e:
            raise CustomException(e, sys)
