from fastapi import FastAPI, WebSocket
from app.api.websocket import handle_whisper_websocket
from app.services.whisper import WhisperService
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

whisper_service = WhisperService(
    model_size=settings.MODEL_SIZE, 
    device=settings.DEVICE
)

@app.get("/")
async def root():
    return {"status": "online", "model": settings.MODEL_SIZE}

@app.websocket("/ws/whisper")
async def websocket_route(websocket: WebSocket):
    await handle_whisper_websocket(websocket, whisper_service)