from fastapi import FastAPI
from api.chat import ChatAPI
from api.chat import ChatType
from api.models import GemmaModel
from api.models import RAGModel 

app = FastAPI()
gemma = GemmaModel()
rag = RAGModel()
app.include_router(ChatAPI(ChatType.FINETUNE, gemma)._router)
app.include_router(ChatAPI(ChatType.RAG, rag)._router)