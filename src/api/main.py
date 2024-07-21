from fastapi import FastAPI
from api.chat import ChatAPI
from api.chat import ChatType
from api.models import GemmaModel_MLX

app = FastAPI()
model = GemmaModel_MLX()
app.include_router(ChatType.FINETUNE, ChatAPI(model)._router)

# app.include_router(ChatType.RAG, ChatAPI(model)._router)
