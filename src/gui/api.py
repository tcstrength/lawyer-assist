# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import requests
import urllib
from enum import Enum
from util import config

class ModelArch(Enum):
    FINETUNE = "Gemma-2B"
    RAG = "RAG"
    
def post_chat_stream(user_input: str, history: list, model: str):
    s = requests.Session()
    if model == ModelArch.FINETUNE.value:
        url = urllib.parse.urljoin(config.LAWYER_API_URL, "/chat/finetune")
    else:
        url = urllib.parse.urljoin(config.LAWYER_API_URL, "/chat/rag")

    try:
    # Open the streaming connection
        payload = {
            "user_input": user_input,
            "history": history,
            "max_tokens": 1024
        }
        print(payload)
        with s.post(url, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses
            for chunk in response.iter_content(chunk_size=1024):
                text = chunk.decode('utf-8')
                yield text
    except requests.RequestException as e:
        yield f"An error occurred while streaming data: {e}"

if __name__ == "__main__":
    for text in post_chat_stream(
        user_input="Mức phạt tiền đối với tổ chức tín dụng không thực hiện đúng quy định về thời hạn đi vay là bao nhiêu?", 
        history=[
            {"role": "user", "content": "Tôi là ai?"},
            {"role": "assistant", "content": "Tôi không biết!"}
        ],
        model=ModelArch.FINETUNE.value):
        print(text, end="")
