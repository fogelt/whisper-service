from faster_whisper import WhisperModel
import numpy as np

class WhisperService:
    def __init__(self, model_size="small", device="cpu"):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

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