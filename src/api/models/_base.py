from datetime import datetime
from abc import abstractmethod
from typing import List, Dict
from pydantic import BaseModel as PydanticModel
from dotenv import load_dotenv
load_dotenv()

class StreamResponse(PydanticModel):
    response_id: str
    content: str
    created_at: int

class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def generate(
        self,
        user_input: str,
        max_tokens: int = 1024,
        history: List[Dict[str, str]] = []
    ) -> str:
        pass