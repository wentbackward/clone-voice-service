# clone-voice-service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://github.com/wentbackward/clone-voice-service/pkgs/container/clone-voice-service)
[![Platform](https://img.shields.io/badge/platform-amd64%20%7C%20arm64-lightgrey)]()
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20CPU-green?logo=nvidia)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://python.org)

Voice cloning TTS + speech-to-text in a single, self-contained API. Powered by [F5-TTS](https://github.com/SWivid/F5-TTS) and [OpenAI Whisper](https://github.com/openai/whisper).

## Features

- **TTS**: Clone any voice from a 5-15 second reference clip. Sub-second generation.
- **STT**: Transcribe audio in 100+ languages. Accepts any audio format.
- Single container, single port, two endpoints
- GPU (CUDA) or CPU — auto-detected at startup
- OGG Opus output for direct use as Telegram/WhatsApp voice messages
- Hot-reload voices — drop files in `voices/`, no restart needed

## Prerequisites

- Docker (with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU)
- Or Docker on any system for CPU-only mode

## Quick Start

```bash
git clone https://github.com/wentbackward/clone-voice-service.git
cd clone-voice-service
./start.sh
```

The script auto-detects GPU availability. Models download on first request (~2GB for TTS, ~1.5GB for Whisper turbo).

### Using Prebuilt Images

Prebuilt multi-arch images are available — no local build required:

```bash
# GPU (NVIDIA) — amd64 or arm64
docker compose up -d

# CPU-only — amd64 or arm64 (Raspberry Pi, etc.)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

| Tag | Platform | GPU | Use case |
|-----|----------|-----|----------|
| `ghcr.io/wentbackward/clone-voice-service:cuda` | amd64 | NVIDIA CUDA | Desktops, servers, WSL |
| `ghcr.io/wentbackward/clone-voice-service:cpu` | amd64, arm64 | None | Raspberry Pi, any system |
| `ghcr.io/wentbackward/clone-voice-service:cpu-arm64` | arm64 | None | Raspberry Pi |
| `ghcr.io/wentbackward/clone-voice-service:cpu-amd64` | amd64 | None | x86 without GPU |

For **NVIDIA arm64** (DGX Spark, Jetson): build locally with `Dockerfile.spark` — see [ARM64 CUDA](#arm64-cuda-dgx-spark-jetson) below.

Also available on Docker Hub: `wentbackward/clone-voice-service`

### ARM64 CUDA (DGX Spark, Jetson)

PyTorch CUDA images are x86-only, so arm64 CUDA builds must be done locally using `Dockerfile.spark`:

```bash
docker build -f Dockerfile.spark -t clone-voice:spark .
./run.sh
```

## Usage

### Text-to-Speech

```bash
curl -X POST http://localhost:3030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?", "voice": "paul", "quality": "fast"}' \
  -o output.ogg
```

### Speech-to-Text

```bash
curl -X POST http://localhost:3030/stt \
  -F "audio=@recording.ogg" \
  -F "format=text"
```

### OpenAI-Compatible STT

Drop-in replacement for the OpenAI transcription API:

```bash
curl -X POST http://localhost:3030/v1/audio/transcriptions \
  -F "file=@recording.ogg" \
  -F "model=whisper-1" \
  -F "response_format=text"
```

Supports `json`, `text`, `verbose_json`, `srt`, and `vtt` response formats.

### List Voices

```bash
curl http://localhost:3030/voices
```

### OpenAI-Compatible TTS

Drop-in replacement for the OpenAI TTS API — works with any client that supports a custom base URL:

```bash
curl -X POST http://localhost:3030/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "tts-1", "input": "Hello!", "voice": "paul"}' \
  -o output.opus
```

**OpenClaw config** — add this to your OpenClaw `settings.json` under `messages`:

```json
{
  "messages": {
    "tts": {
      "auto": "inbound",
      "provider": "openai",
      "openai": {
        "baseUrl": "http://localhost:3030/v1",
        "apiKey": "not-needed",
        "model": "tts-1"
      },
      "maxTextLength": 4000
    }
  }
}
```

Set `baseUrl` to wherever your clone-voice-service is running (e.g. `http://spark-01:3030/v1`). The `voice` field defaults to `"paul"` — to use a different voice, configure it in OpenClaw's voice settings.

See [SKILL.md](SKILL.md) for the full API reference.

## Adding Voices

```
voices/
  paul.wav          # 5-15s reference audio
  paul.txt          # Exact transcript of the audio
  mycustomvoice.wav
  mycustomvoice.txt
```

Tips:
- 5-15 seconds of clear speech, minimal background noise
- 24kHz 16-bit mono WAV is ideal (other formats auto-converted)
- Transcript must match the spoken words exactly
- Try putting an elipsis at the start and end of the transcription if you find you're getting bleed.
- The generated speech speed matches the reference audio's natural pace. For faster or slower output, speed up or slow down the reference WAV file itself.

## Configuration

Edit `config.yaml`:

```yaml
host: "0.0.0.0"
port: 3030
device: "auto"          # "auto", "cuda", or "cpu"

tts:
  model: "F5TTS_v1_Base"
  defaults:
    quality: "quality"  # "fast" or "quality"
    format: "ogg"
    speed: 1.0
    cfg_strength: 2.0

stt:
  backend: "whisper"    # "whisper" (openai-whisper) or "faster-whisper"
  model: "turbo"        # tiny, base, small, medium, large, turbo
  language: null        # null = auto-detect
  defaults:
    format: "json"      # "json", "text", "verbose"
```

The `faster-whisper` backend uses CTranslate2 and is ~4x faster than openai-whisper. To enable it, set `stt.backend: "faster-whisper"` in `config.yaml`.

### CPU-Only Deployment

Set `device: "cpu"` in `config.yaml`. The start script auto-detects and uses a CPU-only Docker image, or force it:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

Consider using a smaller Whisper model (`tiny` or `base`) on CPU.

### Port Override

```bash
VOICE_PORT=8080 ./start.sh
```

## Performance

| Operation | GPU (first) | GPU (warm) | CPU (warm) |
|-----------|------------|------------|------------|
| TTS (short text, fast) | ~30s | <1s | ~5-10s |
| STT (10s audio) | ~15s | ~1-2s | ~10-30s |

GPU memory: ~1.2GB (TTS) + ~1.5GB (Whisper turbo). Models load on first use.

## Architecture

```
              ┌─────────────────────────────────┐
              │        FastAPI (uvicorn)         │
              │                                  │
  POST /tts → │  F5-TTS inference → ffmpeg  → audio
  POST /stt → │  ffmpeg → Whisper inference → json
  GET /voices │  scan voices/ directory          │
  GET /health │  status + device info            │
              └─────────────────────────────────┘
```

## Drawbacks
- Multi-users might not be a good experience. This is a very simple service, locks and queueing are in place as it can only run inference on 1 request at a time. It will handle multiple users / parallel requests but if there's too many simultaneous requests will bog down quickly. Hey! It's free, fire up another docker.
- It's not real-time performance and it's batch mode, you pass the whole text and wait for the whole sound file.
- You cannot control inflection, other than through punctuation. You get what you're given from the enginer!

## License

This project is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Paul Gresham Advisory LLC.

Dependencies:
- F5-TTS: [CC-BY-NC-4.0](https://github.com/SWivid/F5-TTS/blob/main/LICENSE)
- Whisper: [MIT](https://github.com/openai/whisper/blob/main/LICENSE)
