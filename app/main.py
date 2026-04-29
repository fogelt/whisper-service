from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.transcribe import router as transcribe_router

app = FastAPI(title="Whisper Swedish API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe_router, tags=["Transcription"])

@app.get("/")
async def root():
    return {"status": "online"}