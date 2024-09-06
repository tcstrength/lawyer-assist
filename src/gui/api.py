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

class EmbeddingModel(Enum):
    AllMiniLML6v2 = 'sentence-transformers/all-MiniLM-L6-v2'
    HALONG = 'hiieu/halong_embedding'
    
def post_chat_stream(user_input: str, embedding_model_name: str):

    s = requests.Session()
    url = urllib.parse.urljoin(config.LAWYER_API_URL, "/chat")
    embedding_model_id = EmbeddingModel._member_map_[embedding_model_name].value
    try:
    # Open the streaming connection
        payload = {
            "user_input": user_input,
            "max_tokens": 1024,
            "embedding_model_id": embedding_model_id
        }
        print(payload)
        with s.post(url, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses
            for chunk in response.iter_content(chunk_size=1024):
                text = chunk.decode('utf-8', errors='replace')
                yield text
    except requests.RequestException as e:
        yield f"An error occurred while streaming data: {e}"


def get_relevant_documents(query, limit, model_id):
    url = urllib.parse.urljoin(config.LAWYER_API_URL, "/retrieval")
    try:
        response = requests.post(url, params={"query": query, "limit": limit, "model_id": model_id})
        if response.status_code == 200:
            response_json = response.json()
            return [doc['payload']['content'] for doc in response_json]
        else:
            raise Exception(f"Error switching model: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")

def get_gpu_status():
    url = urllib.parse.urljoin(config.LAWYER_API_URL, "/gpu-status")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"gpu_available": False, "gpu_name": "None"}
    except Exception as e:
        print(f"Error fetching GPU status: {e}")
        return {"gpu_available": False, "gpu_name": "Error"}

def get_model():
    url = urllib.parse.urljoin(config.LAWYER_API_URL, "/model")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text

    except Exception as e:
        print(f"Error fetching model: {e}")

if __name__ == "__main__":
    for text in post_chat_stream(
        user_input="Mức phạt tiền đối với tổ chức tín dụng không thực hiện đúng quy định về thời hạn đi vay là bao nhiêu?", 
        history=[
            {"role": "user", "content": "Tôi là ai?"},
            {"role": "assistant", "content": "Tôi không biết!"}
        ],
        model=ModelArch.FINETUNE_GEMA.value):
        print(text, end="")
