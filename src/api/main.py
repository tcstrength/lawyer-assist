from fastapi import FastAPI
from api.chat import ChatAPI
from api.chat import ChatType
from api.models import GemmaModel

app = FastAPI()
model = GemmaModel()
app.include_router(ChatAPI(ChatType.FINETUNE, model)._router)