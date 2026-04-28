import io
import numpy as np
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

#Chance to device="cuda" for GPU
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.")

@app.websocket("/ws/whisper")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")
    
    header_chunk = None
    data_chunks = []
    
    try:
        while True:
            data = await websocket.receive_bytes()

            if header_chunk is None:
                header_chunk = data
                print(f"[WS] EBML Header captured ({len(data)} bytes)")
                continue

            data_chunks.append(data)
            
            current_data_size = sum(len(c) for c in data_chunks)
            

            if current_data_size > 65000:
                try:
                    full_blob = header_chunk + b"".join(data_chunks)
                    audio_file = io.BytesIO(full_blob)
                    
                    #Decode using pydub (requires ffmpeg/ffprobe)
                    #Standardize audio: 16kHz, Mono, 16-bit
                    audio = AudioSegment.from_file(audio_file, format="webm")
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    
                    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                    
                    segments, info = model.transcribe(
                        samples,
                        beam_size=5,
                        language="sv",
                        vad_filter=True,
                        initial_prompt="Detta är en konversation på svenska."
                    )
                    
                    text = " ".join([s.text for s in segments]).strip()

                    if text:
                        print(f"[WHISPER] ({info.language_probability:.2f}): {text}")
                        await websocket.send_json({"text": text})
                    
                    data_chunks = [] 
                    
                except Exception as e:
                    print(f"[DEBUG] Buffer incomplete, waiting for more data... {e}")

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[SERVER ERROR] {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)