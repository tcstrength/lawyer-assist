from fastapi import FastAPI
from api.chat import ChatAPI
from api.models import GemmaModel_MLX

app = FastAPI()
model = GemmaModel_MLX()
app.include_router(ChatAPI(model)._router)
