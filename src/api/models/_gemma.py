# -*- coding: utf-8 -*-
"""
%cd ../../
Author: tcstrength
Date: 2024-07-20
"""

import mlx_lm
from typing import List, Dict
from loguru import logger
from api.models._base import BaseModel


MODEL_ID = "tcstrength/gemma-2b-lawyer-assist"

class GemmaModel_MLX(BaseModel):
    def __init__(self):
        self._model, self._tokenizer = mlx_lm.load(MODEL_ID)
        logger.info(f"Model loaded: {MODEL_ID}")
    
    def generate(
        self,
        user_input: str,
        max_tokens: int = 1024,
        history: List[Dict[str, str]] = []
    ):
        history = history + [
            {"role": "user", "content": user_input },
        ]

        prompt = self._tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        print("Prompt:", prompt)
        return mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens
        )


if __name__ == "__main__":
    self = GemmaModel_MLX()
    self._tokenizer.apply_chat_template([
        {"role": "user", "content": "What the hell!"}
    ], tokenize=False)