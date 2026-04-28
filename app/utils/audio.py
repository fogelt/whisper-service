import io
import numpy as np
from pydub import AudioSegment

def process_webm_chunk(header: bytes, chunks: list):
    full_blob = header + b"".join(chunks)
    audio_file = io.BytesIO(full_blob)
    
    audio = AudioSegment.from_file(audio_file, format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    
    # Convert to float32
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples