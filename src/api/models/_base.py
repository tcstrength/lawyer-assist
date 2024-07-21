from abc import abstractmethod
from typing import List, Dict

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