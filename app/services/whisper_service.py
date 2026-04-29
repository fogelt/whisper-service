from faster_whisper import WhisperModel
from app.core.config import settings
import numpy as np

class WhisperService:
    def __init__(self, model_id=settings.MODEL_ID, device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE):
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(self, audio_samples: np.ndarray):
        segments, info = self.model.transcribe(
            audio_samples,
            beam_size=5,
            language="sv",
            vad_filter=True,
            initial_prompt="Detta är en konversation på svenska.",
            condition_on_previous_text=False
        )
        text = " ".join([s.text for s in segments]).strip()
        return text, info