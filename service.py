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
import logging
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

log = logging.getLogger("voice-service")
logging.basicConfig(level=getattr(logging, CONFIG.get("log_level", "INFO").upper(), logging.INFO))


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
_tts_infer_lock = threading.Lock()  # serialize inference — F5-TTS is not thread-safe

_stt_model = None
_stt_lock = threading.Lock()
_stt_infer_lock = threading.Lock()  # serialize inference — Whisper is not thread-safe


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


def _patch_f5tts():
    """Patch F5-TTS internals for stability.

    1. Disable internal ThreadPoolExecutor — the DiT transformer caches text
       embeddings as instance variables, so concurrent batch processing
       corrupts the cache causing "Sizes of tensors must match" errors.
    2. Enforce a minimum generation duration — short texts get allocated too
       few frames, causing truncated audio output.
    """
    import f5_tts.infer.utils_infer as f5_utils

    # --- Patch 1: Sequential executor ---
    class _SequentialExecutor:
        """Drop-in replacement that runs submissions sequentially."""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future
            f = Future()
            try:
                f.set_result(fn(*args, **kwargs))
            except Exception as e:
                f.set_exception(e)
            return f

    f5_utils.ThreadPoolExecutor = _SequentialExecutor

    # --- Patch 2: Consistent speed across all text lengths ---
    # F5-TTS overrides speed to 0.3 for texts < 10 bytes, causing
    # inconsistent speech rate. We patch _infer_basic to always use the
    # caller's speed, and enforce a minimum duration floor to prevent
    # truncation on very short texts.
    _orig_infer_batch = f5_utils.infer_batch_process

    def _patched_infer_batch(*args, **kwargs):
        return _orig_infer_batch(*args, **kwargs)

    # Patch the inner function that overrides speed
    import types
    _orig_module_source = f5_utils.__name__

    _real_infer_basic = None

    def _patch_inner_speed():
        """Remove speed=0.3 override and add minimum duration floor."""
        import f5_tts.infer.utils_infer as mod
        orig_ibp = mod.infer_batch_process

        def patched_ibp(
            ref_audio, ref_text, gen_text_batches, model_obj, vocoder,
            mel_spec_type="vocos", progress=None, target_rms=0.1,
            cross_fade_duration=0.15, nfe_step=32, cfg_strength=2.0,
            sway_sampling_coef=-1, speed=1, fix_duration=None,
            device=None, streaming=False, chunk_size=2048,
        ):
            import torchaudio
            audio, sr = ref_audio
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != mod.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, mod.target_sample_rate)
                audio = resampler(audio)
            audio = audio.to(device)

            generated_waves = []
            spectrograms = []

            if len(ref_text[-1].encode("utf-8")) == 1:
                ref_text = ref_text + " "

            hop_length = mod.hop_length
            ref_audio_len = audio.shape[-1] // hop_length

            def _infer_consistent(gen_text):
                text_list = [ref_text + gen_text]
                final_text_list = mod.convert_char_to_pinyin(text_list)

                if fix_duration is not None:
                    duration = int(fix_duration * mod.target_sample_rate / hop_length)
                else:
                    ref_text_len = len(ref_text.encode("utf-8"))
                    gen_text_len = len(gen_text.encode("utf-8"))
                    # Always use caller's speed — no override for short texts
                    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
                    # Minimum 1.5s of generated audio to prevent truncation
                    min_gen_frames = int(1.5 * mod.target_sample_rate / hop_length)
                    if (duration - ref_audio_len) < min_gen_frames:
                        duration = ref_audio_len + min_gen_frames

                with torch.inference_mode():
                    generated, _ = model_obj.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )
                    del _
                    generated = generated.to(torch.float32)
                    generated = generated[:, ref_audio_len:, :]
                    generated = generated.permute(0, 2, 1)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(generated)
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(generated)
                    if rms < target_rms:
                        generated_wave = generated_wave * rms / target_rms
                    generated_wave = generated_wave.squeeze().cpu().numpy()

                generated_cpu = generated[0].cpu().numpy()
                del generated
                return generated_wave, generated_cpu

            if streaming:
                for gen_text in (progress.tqdm(gen_text_batches) if progress else gen_text_batches):
                    wave, _ = _infer_consistent(gen_text)
                    for j in range(0, len(wave), chunk_size):
                        yield wave[j:j+chunk_size], mod.target_sample_rate
            else:
                for gen_text in (progress.tqdm(gen_text_batches) if progress else gen_text_batches):
                    generated_wave, generated_mel_spec = _infer_consistent(gen_text)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

                if generated_waves:
                    if cross_fade_duration <= 0:
                        final_wave = np.concatenate(generated_waves)
                    else:
                        final_wave = generated_waves[0]
                        for i in range(1, len(generated_waves)):
                            prev_wave = final_wave
                            next_wave = generated_waves[i]
                            cross_fade_samples = int(cross_fade_duration * mod.target_sample_rate)
                            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
                            if cross_fade_samples <= 0:
                                final_wave = np.concatenate([prev_wave, next_wave])
                                continue
                            fade_out = np.linspace(1, 0, cross_fade_samples)
                            fade_in = np.linspace(0, 1, cross_fade_samples)
                            cross_faded = prev_wave[-cross_fade_samples:] * fade_out + next_wave[:cross_fade_samples] * fade_in
                            final_wave = np.concatenate([prev_wave[:-cross_fade_samples], cross_faded, next_wave[cross_fade_samples:]])

                    combined_spectrogram = np.concatenate(spectrograms, axis=1)
                    yield final_wave, mod.target_sample_rate, combined_spectrogram
                else:
                    yield None, mod.target_sample_rate, None

        mod.infer_batch_process = patched_ibp

    _patch_inner_speed()

_patch_f5tts()


def generate_audio(text, ref_file, ref_text, nfe_step=32,
                   cfg_strength=2.0, speed=1.0, seed=None):
    with _tts_infer_lock:
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
    matched = req.voice if req.voice in voices else None
    if matched:
        v = voices[matched]
    elif voices:
        matched = next(iter(voices))
        v = voices[matched]
    else:
        raise HTTPException(500, "No voices configured")
    log.info(f"openai-tts voice={req.voice!r} → {matched!r}, fmt={req.response_format!r}, len={len(req.input)}")

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
            with _stt_infer_lock:
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
