import os
import dotenv
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

dotenv.load_dotenv(os.path.join(get_project_root(), "/.env"))

PROJECT_ROOT = get_project_root()
DATA_ARTICLE_DIR = os.path.join(PROJECT_ROOT, os.getenv("DATA_ARTICLE_DIR"))
DATA_QNA_DIR = os.path.join(PROJECT_ROOT, os.getenv("DATA_QNA_DIR"))
DATA_SRC_URL_PREFIX = os.getenv("DATA_SRC_URL_PREFIX")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")