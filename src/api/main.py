from enum import Enum
import logging
from typing import Dict, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pydantic
import torch
from api.models import RAGModel
from data.qdrant_helper import QdrantHelper 

app = FastAPI()

logger = logging.getLogger(__file__)


class ModelId(Enum):
    FINETUNE_GEMA = 'tcstrength/gemma-2b-lawyer-assist'
    BASE_GEMA = 'google/gemma-2-2b-it'

class EmbeddingModelId(Enum):
    AllMiniLML6v2 = 'sentence-transformers/all-MiniLM-L6-v2'
    HALONG = 'hiieu/halong_embedding'

class ChatRequest(pydantic.BaseModel):
    user_input: str
    history: List[Dict[str, str]] = []
    max_tokens: int = 1024
    embedding_model_id: str = EmbeddingModelId.HALONG.value

# MODEL_ID = 'tcstrength/gemma-2b-lawyer-assist'
MODEL_ID = 'google/gemma-2-2b-it'

# EMBEDDING_MODEL_ID = 'hiieu/halong_embedding'
# EMBEDDING_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'

# gemma = GemmaModel()
rag = RAGModel(model_id=MODEL_ID)

halong_embedding = QdrantHelper(embedding_dimension=768, collection_name='halong_embedding-collection', embedding_model_id='hiieu/halong_embedding', device='cuda:1')
MiniLM_embedding = QdrantHelper(embedding_dimension=384, collection_name='new_law_collection', embedding_model_id='sentence-transformers/all-MiniLM-L6-v2', device='cuda:1')
# app.include_router(ChatAPI(ChatType.FINETUNE, gemma)._router)
# app.include_router(ChatAPI(rag)._router)

@app.post("/chat")
async def chat(req: ChatRequest):
    logger.info(f"Chat request: {req.user_input}")
    documents = []
    if req.embedding_model_id == EmbeddingModelId.AllMiniLML6v2.value:
        documents = [doc.payload['content'] for doc in MiniLM_embedding.search(req.user_input, 10)]
    
    if req.embedding_model_id == EmbeddingModelId.HALONG.value:
        documents = [doc.payload['content'] for doc in halong_embedding.search(req.user_input, 10)]
    
    stream = rag.generate(
        user_input=req.user_input,
        max_tokens=req.max_tokens,
        history=req.history,
        documents=documents
    )
    return StreamingResponse(stream)

@app.post("/retrieval")
async def get_relevant_documents(query: str, limit: int, model_id: str):
    if model_id == EmbeddingModelId.AllMiniLML6v2.value:
        return MiniLM_embedding.search(query, limit)
    
    if model_id == EmbeddingModelId.HALONG.value:
        return halong_embedding.search(query, limit)

@app.get("/gpu-status")
async def gpu_status():
    return {
        "gpu_available": torch.cuda.is_available(), 
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

@app.get("/model")
async def get_models():
    return MODEL_ID

