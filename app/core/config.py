import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Whisper Swedish API"
    MODEL_SIZE: str = os.getenv("MODEL_SIZE", "small")
    DEVICE: str = os.getenv("DEVICE", "cuda")  # "cuda" or "cpu"
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "int8")
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    CHUNK_THRESHOLD: int = 65000

    #Parallel workers
    WORKER_COUNT: int = 4

settings = Settings()