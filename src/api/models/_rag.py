from typing import Dict, List
from api.models._base import BaseModel

class RAGModel(BaseModel):

    def generate(
        self,
        user_input: str,
        max_tokens: int = 1024,
        history: List[Dict[str, str]] = []
    ) -> str:
        pass

    def get_relevant_documents(query: str, k: int = 3):
        query_embedding = retriever_model.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]]