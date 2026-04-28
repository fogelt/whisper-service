import io
import numpy as np
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

# Device="cuda" if you have an NVIDIA GPU, otherwise "cpu"
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded successfully.")

@app.websocket("/ws/whisper")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")
    
    header_chunk = None
    data_chunks = []
    last_transcript = "" # Track previous text for the prompt logic
    
    try:
        while True:
            data = await websocket.receive_bytes()

            if header_chunk is None:
                header_chunk = data
                print(f"[WS] EBML Header captured ({len(data)} bytes)")
                continue

            data_chunks.append(data)
            current_data_size = sum(len(c) for c in data_chunks)
            
            # Process every ~2 seconds (approx 65KB+)
            if current_data_size > 65000:
                try:
                    # Reconstruct the WebM file
                    full_blob = header_chunk + b"".join(data_chunks)
                    audio_file = io.BytesIO(full_blob)
                    
                    # Decode and standardize
                    audio = AudioSegment.from_file(audio_file, format="webm")
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    
                    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                    
                    # Transcribe with Swedish settings
                    segments, info = model.transcribe(
                        samples,
                        beam_size=5,
                        language="sv",
                        initial_prompt="Detta är en konversation på svenska." if not last_transcript else None, 
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500), 
                        condition_on_previous_text=False 
                    )

                    new_text = " ".join([s.text for s in segments]).strip()

                    hallucinations = [
                        "Detta är en konversation på svenska."
                    ]

                    if new_text and new_text not in hallucinations:
                        last_transcript = new_text
                        await websocket.send_json({"text": new_text})
                    
                    data_chunks = data_chunks[-2:] 
                    
                except Exception as e:
                    print(f"[DEBUG] Buffer incomplete, waiting for more data...")

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[SERVER ERROR] {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)