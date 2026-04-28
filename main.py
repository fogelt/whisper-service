import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI()

#använd device="cuda" för GPU
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.websocket("/ws/whisper")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            if len(audio_buffer) > 32000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                
                segments, _ = model.transcribe(audio_np, beam_size=5)
                text = "".join([s.text for s in segments]).strip()

                if text:
                    await websocket.send_json({"text": text, "is_final": False})
                
                audio_buffer.clear()

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)