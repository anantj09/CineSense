import logging
import os
from datetime import datetime

from src.utils.common import read_yaml
paths_config = read_yaml("configs/paths_config.yaml")
LOG_DIR = paths_config["logs_dir"]

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("IMDbSentimentLogger")