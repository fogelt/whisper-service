import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File
from app.core.config import settings
from app.services.whisper_service import WhisperService
from app.utils.audio import process_webm_chunk

router = APIRouter()
whisper = WhisperService()
executor = ThreadPoolExecutor(max_workers=settings.WORKER_COUNT)

def sync_transcribe(audio_bytes):
    samples = process_webm_chunk(audio_bytes, [])
    text, info = whisper.transcribe(samples)
    return {"text": text, "language": info.language}

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...), 
    sequence_id: int = 0
):
    audio_bytes = await file.read()
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, sync_transcribe, audio_bytes)
    
    return {
        "text": result["text"],
        "language": result["language"],
        "sequence_id": sequence_id
    }