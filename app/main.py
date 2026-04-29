from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.transcribe import router as transcribe_router

app = FastAPI(title="Whisper Swedish API")

# 1. Define allowed origins
# For "CORS *", you use ["*"]
origins = ["*"]

# 2. Add the middleware to the app instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers (Content-Type, etc.)
)

app.include_router(transcribe_router, tags=["Transcription"])

@app.get("/")
async def root():
    return {"status": "online"}