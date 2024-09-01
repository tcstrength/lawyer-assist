# -*- coding: utf-8 -*-
"""
Author: tcstrength
Date: 2024-07-04
"""

import pydantic
from typing import List
from typing import Dict 
from enum import Enum
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import BaseModel


class ChatType(Enum):
    FINETUNE = "finetune"
    RAG = "rag"


class ChatRequest(pydantic.BaseModel):
    user_input: str
    history: List[Dict[str, str]] = []
    max_tokens: int = 1024


class ChatAPI:
    def __init__(self, type: ChatType, model: BaseModel):
        self._model = model
        self._type = type
        self._router = APIRouter()
        self._router.add_api_route(f"/chat/{type.value}", self.chat, methods=["POST"])

    async def chat(self, req: ChatRequest):
        stream = self._model.generate(
            user_input=req.user_input,
            max_tokens=req.max_tokens,
            history=req.history
        )
        return StreamingResponse(stream)