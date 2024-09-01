# -*- coding: utf-8 -*-
"""
%cd ../../
Author: tcstrength
Date: 2024-07-20
"""

from typing import List, Dict
from loguru import logger
from api.models._base import BaseModel

# LoRA
MODEL_ID = "tcstrength/gemma-2b-lawyer-assist"

try:
    import mlx_lm
    class GemmaModel(BaseModel):
        def __init__(self, model_id: str = MODEL_ID):
            self._model, self._tokenizer = mlx_lm.load(model_id)
            logger.info(f"Model loaded: {model_id}")
        
        def generate(
            self,
            user_input: str,
            max_tokens: int = 1024,
            history: List[Dict[str, str]] = []
        ):
            history = history + [
                {"role": "user", "content": user_input}
            ]

            prompt = self._tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )

            # print("Prompt:", prompt)
            return mlx_lm.stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
except Exception as e:
    logger.info(e)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import TextIteratorStreamer
    class GemmaModel(BaseModel):
        def __init__(self, model_id: str = MODEL_ID):
            self._tokenizer = AutoTokenizer.from_pretrained("tcstrength/gemma-2b-lawyer-assist")
            self._model = AutoModelForCausalLM.from_pretrained("tcstrength/gemma-2b-lawyer-assist")
            logger.info(f"Model loaded: {model_id}")
        
        def generate(
            self,
            user_input: str,
            max_tokens: int = 1024,
            history: List[Dict[str, str]] = []
        ):
            history = history + [
                {"role": "user", "content": user_input },
            ]

            input_ids = self._tokenizer.apply_chat_template(
                history, add_generation_prompt=True,
                return_tensors="pt"
            )

            streamer = TextIteratorStreamer(self._tokenizer)
            generation_kwargs = {
                "input_ids": input_ids,
                "max_length": max_tokens,
                "temperature": 0.0,
                "streamer": streamer,
            }
            import threading
            thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text


if __name__ == "__main__":
    self = GemmaModel()
    user_input = "Mức phạt tiền đối với tổ chức tín dụng không thực hiện đúng quy định về thời hạn đi vay là bao nhiêu?"
    for text in self.generate(user_input):
        print(text, end="")
