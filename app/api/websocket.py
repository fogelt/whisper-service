from fastapi import WebSocket, WebSocketDisconnect
from app.services.whisper import WhisperService
from app.utils.audio import process_webm_chunk
from app.core.config import settings

async def handle_whisper_websocket(websocket: WebSocket, whisper_service: WhisperService):
    await websocket.accept()
    header_chunk = None
    data_chunks = []
    last_transcript = ""

    try:
        while True:
            data = await websocket.receive_bytes()

            if header_chunk is None:
                header_chunk = data
                continue

            data_chunks.append(data)
            
            if sum(len(c) for c in data_chunks) > settings.CHUNK_THRESHOLD:
                try:
                    samples = process_webm_chunk(header_chunk, data_chunks)
                    text, info = whisper_service.transcribe(samples)

                    # Hallucination filter
                    hallucinations = ["Välkomna!", "Detta är en konversation på svenska."]
                    
                    if text and text not in hallucinations:
                        await websocket.send_json({"text": text, "lang": info.language})
                        last_transcript = text
                    
                    # Sliding window: keep the tail for continuity
                    data_chunks = data_chunks[-2:]
                except Exception:
                    # Usually means the audio chunk was cut mid-frame, wait for next
                    pass
    except WebSocketDisconnect:
        print("[WS] Client disconnected")