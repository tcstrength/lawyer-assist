import os
import dotenv
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

dotenv.load_dotenv(os.path.join(get_project_root(), "/.env"))

PROJECT_ROOT = get_project_root()
DATA_ARTICLE_DIR = os.path.join(PROJECT_ROOT, os.getenv("DATA_ARTICLE_DIR", "data/article"))
DATA_QNA_DIR = os.path.join(PROJECT_ROOT, os.getenv("DATA_QNA_DIR", "data/qna"))
DATA_SRC_URL_PREFIX = os.getenv(
    "DATA_SRC_URL_PREFIX", "https://thuvienphapluat.vn/hoi-dap-phap-luat/tien-te-ngan-hang"
)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LAWYER_API_URL = os.getenv("LAWYER_API_URL")
MODEL_VECTORIZER_ID = os.getenv("MODEL_VECTORIZER_ID")
MODEL_CHAT_FT_ID = os.getenv("MODEL_CHAT_FT_ID")