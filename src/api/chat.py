# -*- coding: utf-8 -*-
"""
Author: tcstrength
Date: 2024-07-04
"""

from enum import Enum
import logging
import pydantic
from typing import List
from typing import Dict 
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import RAGModel

logger = logging.getLogger(__file__)

class ChatRequest(pydantic.BaseModel):
    user_input: str
    history: List[Dict[str, str]] = []
    max_tokens: int = 1024

class ModelId(Enum):
    FINETUNE_GEMA = 'tcstrength/gemma-2b-lawyer-assist'
    BASE_GEMA = 'google/gemma-2-2b-it'

class EmbeddingModelId(Enum):
    AllMiniLML6v2 = 'sentence-transformers/all-MiniLM-L6-v2'
    HALONG = 'hiieu/halong_embedding'

class ChatAPI:
    def __init__(self, model: RAGModel):
        self._model = model
        self._router = APIRouter()
        self._router.add_api_route(f"/chat", self.chat, methods=["POST"])
        self._router.add_api_route(f"/model", self.change_model, methods=["POST"])
        self._router.add_api_route(f"/embedding", self.change_embedding_model, methods=["POST"])
        self._router.add_api_route(f"/retrieval", self.get_relevant_documents, methods=["POST"])

    async def chat(self, req: ChatRequest):
        logger.info(f"Chat request: {req.user_input}")
        stream = self._model.generate(
            user_input=req.user_input,
            max_tokens=req.max_tokens,
            history=req.history
        )
        return StreamingResponse(stream)
    
    async def change_model(self, model_id: str):
        logger.info(f"Change model model: {model_id}")
        if model_id in ModelId._member_map_.values():
            self._model.change_model(model_id)
            return {"success": True}
        else:
            return {"error": f"Don't support the model {model_id}."}
        
    async def change_embedding_model(self, model_id: str):
        logger.info(f"Change embedding model: {model_id}")
        if model_id in EmbeddingModelId._member_map_.values():
            self._model.change_embedding_model(model_id)
            return {"success": True}
        else:
            return {"error": f"Don't support the model {model_id}."}
    
    async def get_relevant_documents(self, query: str, limit: int):
        return self._model.get_relevant_documents(query, limit)