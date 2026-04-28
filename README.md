# Whisper Service

Serverless speech-to-text API för svenska, baserad på OpenAI Whisper. WebSocket-baserad streaming-transkribering med låg latens.

## Krav

- Python 3.12+
- FFmpeg
- CUDA (valfritt, för GPU-acceleration)

## Lokal setup

```bash
# Installation
pip install -e .

# Eller med uv
uv sync

# Starta server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Servern körs på `http://localhost:8001`

## Konfiguration

Miljövariabler:

| Variabel | Standard | Beskrivning |
|----------|----------|------------|
| `MODEL_SIZE` | `small` | Modellstorlek: `tiny`, `small`, `medium`, `large` |
| `DEVICE` | `cpu` | `cpu` eller `cuda` |
| `COMPUTE_TYPE` | `int8` | `int8`, `int16` eller `float32` |
| `CHUNK_THRESHOLD` | `65000` | Bytes före transkribering |

Exempel för GPU:
```bash
MODEL_SIZE=large DEVICE=cuda COMPUTE_TYPE=float32 uv run uvicorn app.main:app
```

## API

### WebSocket `/ws/whisper`

Streaming WebM-audio → transkriberad text

**Request**: WebM-audio frames
**Response**: `{"text": "...", "lang": "sv"}`

### GET `/`

Hälsostatus och modellinfo

## Docker

```bash
# Build
docker build -t whisper-backend:v2 .

# Run
docker run -p 8001:8001 -e DEVICE=cpu whisper-backend:v2
```

## Kubernetes

```bash
kubectl apply -f k8s/

# Verifiera
kubectl get pods -l app=whisper
kubectl port-forward service/whisper-backend 8001:8001
```

Resource requests/limits: 1-2 CPU, 2-4 GB RAM (CPU-mode)

## Development

```bash
# Format & lint
uv run black app/
uv run ruff check app/

# Testing
uv run pytest
```

## Performance

- Modell laddas i minnet vid start (~500MB för `small`)
- Sliding window transkribering för kontinuerlighet
- VAD-filter för brusreduktion
- Hallucination-filter för standardfraser