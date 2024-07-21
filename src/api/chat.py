# -*- coding: utf-8 -*-
"""
Author: tcstrength
Date: 2024-07-04
"""

import pydantic
from typing import List
from typing import Tuple
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import BaseModel

class ChatRequest(pydantic.BaseModel):
    user_input: str
    history: List[Tuple[str, str]] = []


class ChatAPI:
    def __init__(self, model: BaseModel):
        self._model = model
        self._router = APIRouter()
        self._router.add_api_route("/chat", self.chat, methods=["POST"])

    async def chat(self, req: ChatRequest):
        stream = self._model.generate(req.user_input, max_tokens=512)
        return StreamingResponse(stream)