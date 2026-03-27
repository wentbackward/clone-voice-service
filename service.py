#!/usr/bin/env python3
"""
Clone Voice Service — TTS (F5-TTS) + STT (Whisper) in a single API.

  POST /tts               — text + voice → audio
  POST /v1/audio/speech   — OpenAI-compatible TTS endpoint
  POST /stt               — audio → text
  GET  /voices
  GET  /health
"""

import io
import os
import tempfile
import subprocess
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from typing import Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("VOICE_CONFIG", "/app/config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

VOICES_DIR = Path(CONFIG["voices_dir"])
TTS_DEFAULTS = CONFIG.get("tts", {}).get("defaults", {})
STT_DEFAULTS = CONFIG.get("stt", {}).get("defaults", {})


def detect_device() -> str:
    setting = CONFIG.get("device", "auto")
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


DEVICE = detect_device()

app = FastAPI(
    title="Clone Voice Service",
    description="Voice cloning TTS + speech-to-text in a single API",
)

# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------
_tts_model = None
_tts_lock = threading.Lock()

_stt_model = None
_stt_lock = threading.Lock()


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        with _tts_lock:
            if _tts_model is None:
                from f5_tts.api import F5TTS
                tts_cfg = CONFIG.get("tts", {})
                _tts_model = F5TTS(
                    model=tts_cfg.get("model", "F5TTS_v1_Base"),
                    device=DEVICE,
                )
    return _tts_model


def get_stt_model():
    global _stt_model
    if _stt_model is None:
        with _stt_lock:
            if _stt_model is None:
                import whisper
                stt_cfg = CONFIG.get("stt", {})
                _stt_model = whisper.load_model(
                    stt_cfg.get("model", "turbo"),
                    device=DEVICE,
                )
    return _stt_model


# ---------------------------------------------------------------------------
# Voice discovery
# ---------------------------------------------------------------------------
def discover_voices() -> dict:
    voices = {}
    if not VOICES_DIR.exists():
        return voices
    for wav_file in sorted(VOICES_DIR.glob("*.wav")):
        name = wav_file.stem
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            voices[name] = {
                "file": str(wav_file),
                "transcript": txt_file.read_text().strip(),
            }
    return voices


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    voice: str
    speed: float = Field(default=None, ge=0.1, le=3.0)
    seed: int = Field(default=None, ge=-1)
    voice_transcript: str | None = Field(default=None)
    quality: str = Field(default=None)
    format: str = Field(default=None)
    cfg_strength: float | None = Field(default=None, ge=0.1, le=5.0)


def generate_audio(text, ref_file, ref_text, nfe_step=32,
                   cfg_strength=2.0, speed=1.0, seed=None):
    model = get_tts_model()
    wav, sr, _ = model.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=text,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        speed=speed,
        seed=seed if seed and seed >= 0 else None,
        show_info=lambda x: None,
    )
    return wav, sr


def wav_to_format(wav, sr, fmt):
    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="FLAC")
        return buf.getvalue(), "audio/flac"

    fmt_map = {"ogg": ("libopus", "audio/ogg"), "mp3": ("libmp3lame", "audio/mpeg")}
    codec, mime = fmt_map.get(fmt, ("libopus", "audio/ogg"))

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")

    with tempfile.NamedTemporaryFile(suffix=".wav") as inf, \
         tempfile.NamedTemporaryFile(suffix=f".{fmt}") as outf:
        inf.write(buf.getvalue())
        inf.flush()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", inf.name, "-c:a", codec, outf.name],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed: {result.stderr.decode()[-200:]}")
        return open(outf.name, "rb").read(), mime


@app.post("/tts", response_class=Response)
async def text_to_speech(req: TTSRequest):
    import asyncio

    quality = req.quality or TTS_DEFAULTS.get("quality", "quality")
    fmt = req.format or TTS_DEFAULTS.get("format", "ogg")
    speed = req.speed if req.speed is not None else TTS_DEFAULTS.get("speed", 1.0)
    seed = req.seed if req.seed is not None else TTS_DEFAULTS.get("seed", -1)
    cfg = req.cfg_strength if req.cfg_strength is not None else TTS_DEFAULTS.get("cfg_strength", 2.0)
    nfe_step = 16 if quality == "fast" else 32

    voices = discover_voices()

    if req.voice in voices:
        v = voices[req.voice]
        ref_file = v["file"]
        ref_text = v["transcript"]
    else:
        ref_file = str(VOICES_DIR / req.voice)
        if not os.path.exists(ref_file):
            raise HTTPException(404, f"Voice not found: {req.voice}")
        ref_text = req.voice_transcript or ""

    try:
        wav, sr = await asyncio.to_thread(
            generate_audio, req.text, ref_file, ref_text,
            nfe_step, cfg, speed, seed,
        )
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {str(e)}")

    audio_bytes, mime = wav_to_format(wav, sr, fmt)
    return Response(content=audio_bytes, media_type=mime)


# ---------------------------------------------------------------------------
# OpenAI-compatible TTS  (POST /v1/audio/speech)
# ---------------------------------------------------------------------------
class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "paul"
    response_format: Optional[str] = "opus"
    speed: Optional[float] = 1.0


OPENAI_FMT_MAP = {
    "mp3": "mp3",
    "opus": "ogg",
    "aac": "mp3",   # ffmpeg can produce aac but we map to mp3 for simplicity
    "flac": "flac",
    "wav": "flac",   # serve lossless
    "pcm": "flac",
}

OPENAI_MIME_MAP = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/mpeg",
    "flac": "audio/flac",
    "wav": "audio/flac",
    "pcm": "audio/flac",
}


@app.post("/v1/audio/speech", response_class=Response)
async def openai_tts(req: OpenAISpeechRequest):
    import asyncio

    voices = discover_voices()
    if req.voice in voices:
        v = voices[req.voice]
    elif voices:
        v = next(iter(voices.values()))
    else:
        raise HTTPException(500, "No voices configured")

    fmt = OPENAI_FMT_MAP.get(req.response_format, "ogg")
    mime = OPENAI_MIME_MAP.get(req.response_format, "audio/ogg")
    speed = req.speed or 1.0

    try:
        wav, sr = await asyncio.to_thread(
            generate_audio, req.input, v["file"], v["transcript"],
            16, 2.0, speed, None,
        )
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {str(e)}")

    audio_bytes, _ = wav_to_format(wav, sr, fmt)
    return Response(content=audio_bytes, media_type=mime)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
@app.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: str = Form(default=None),
    format: str = Form(default=None),
):
    import asyncio

    fmt = format or STT_DEFAULTS.get("format", "json")
    lang = language or CONFIG.get("stt", {}).get("language")

    audio_data = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        # Convert to wav with ffmpeg to handle any input format
        with tempfile.NamedTemporaryFile(suffix=_guess_ext(audio.filename)) as inp:
            inp.write(audio_data)
            inp.flush()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", inp.name, "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", tmp_path],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                os.unlink(tmp_path)
                raise HTTPException(400, f"Could not decode audio: {result.stderr.decode()[-200:]}")

    try:
        def _transcribe():
            model = get_stt_model()
            opts = {}
            if lang:
                opts["language"] = lang
            return model.transcribe(tmp_path, **opts)

        result = await asyncio.to_thread(_transcribe)
    finally:
        os.unlink(tmp_path)

    if fmt == "text":
        return Response(content=result["text"].strip(), media_type="text/plain")
    elif fmt == "verbose":
        return JSONResponse({
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ],
        })
    else:
        return JSONResponse({
            "text": result["text"].strip(),
            "language": result.get("language"),
        })


def _guess_ext(filename: str | None) -> str:
    if filename:
        ext = Path(filename).suffix
        if ext:
            return ext
    return ".wav"


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------
@app.get("/voices")
async def list_voices():
    return {name: {"file": v["file"]} for name, v in discover_voices().items()}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "tts_loaded": _tts_model is not None,
        "stt_loaded": _stt_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG.get("host", "0.0.0.0"), port=CONFIG.get("port", 3030))
