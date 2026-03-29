#!/usr/bin/env python3
"""Test all endpoints of the Clone Voice Service."""

import sys
import time
import requests

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3030"
passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
    except AssertionError as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {type(e).__name__}: {e}")
        failed += 1


print(f"Testing Clone Voice Service at {HOST}")
print("=" * 50)

# Health
print("GET  /health ... ", end="", flush=True)
def t_health():
    r = requests.get(f"{HOST}/health")
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"OK (device={data.get('device', '?')})")
test("health", t_health)

# Voices
print("GET  /voices ... ", end="", flush=True)
def t_voices():
    r = requests.get(f"{HOST}/voices")
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    print(f"OK ({len(data)} voices: {', '.join(data.keys())})")
test("voices", t_voices)

# TTS — ogg fast
print("POST /tts (ogg, fast) ... ", end="", flush=True)
def t_tts_ogg():
    start = time.time()
    r = requests.post(f"{HOST}/tts", json={
        "text": "Hello, this is a test of the text to speech service.",
        "voice": "paul", "quality": "fast", "format": "ogg",
    })
    elapsed = time.time() - start
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    with open("/tmp/test_tts.ogg", "wb") as f:
        f.write(r.content)
    print(f"OK ({len(r.content)} bytes, {elapsed:.1f}s)")
test("tts_ogg", t_tts_ogg)

# TTS — flac quality
print("POST /tts (flac, quality) ... ", end="", flush=True)
def t_tts_flac():
    r = requests.post(f"{HOST}/tts", json={
        "text": "Testing flac output.",
        "voice": "paul", "quality": "quality", "format": "flac",
    })
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    print(f"OK ({len(r.content)} bytes)")
test("tts_flac", t_tts_flac)

# TTS — mp3
print("POST /tts (mp3) ... ", end="", flush=True)
def t_tts_mp3():
    r = requests.post(f"{HOST}/tts", json={
        "text": "Testing mp3 output.",
        "voice": "paul", "format": "mp3",
    })
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    print(f"OK ({len(r.content)} bytes)")
test("tts_mp3", t_tts_mp3)

# TTS — invalid voice
print("POST /tts (bad voice) ... ", end="", flush=True)
def t_tts_bad():
    r = requests.post(f"{HOST}/tts", json={"text": "test", "voice": "nonexistent"})
    assert r.status_code == 404, f"expected 404, got {r.status_code}"
    print("OK (correctly returned 404)")
test("tts_bad_voice", t_tts_bad)

# STT — json
print("POST /stt (json) ... ", end="", flush=True)
def t_stt_json():
    start = time.time()
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/stt", files={"audio": f}, data={"format": "json"})
    elapsed = time.time() - start
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    data = r.json()
    print(f"OK ({elapsed:.1f}s) \"{data['text'][:60]}\"")
test("stt_json", t_stt_json)

# STT — text
print("POST /stt (text) ... ", end="", flush=True)
def t_stt_text():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/stt", files={"audio": f}, data={"format": "text"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    assert r.headers["content-type"].startswith("text/plain")
    print(f"OK \"{r.text[:60]}\"")
test("stt_text", t_stt_text)

# STT — verbose
print("POST /stt (verbose) ... ", end="", flush=True)
def t_stt_verbose():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/stt", files={"audio": f}, data={"format": "verbose"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert "segments" in data
    print(f"OK ({len(data['segments'])} segments)")
test("stt_verbose", t_stt_verbose)

# OpenAI STT — json
print("POST /v1/audio/transcriptions (json) ... ", end="", flush=True)
def t_oai_stt_json():
    start = time.time()
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/v1/audio/transcriptions",
                          files={"file": f}, data={"model": "whisper-1"})
    elapsed = time.time() - start
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    data = r.json()
    assert "text" in data
    print(f"OK ({elapsed:.1f}s) \"{data['text'][:60]}\"")
test("oai_stt_json", t_oai_stt_json)

# OpenAI STT — text
print("POST /v1/audio/transcriptions (text) ... ", end="", flush=True)
def t_oai_stt_text():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/v1/audio/transcriptions",
                          files={"file": f}, data={"response_format": "text"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    assert r.headers["content-type"].startswith("text/plain")
    print(f"OK \"{r.text[:60]}\"")
test("oai_stt_text", t_oai_stt_text)

# OpenAI STT — verbose_json
print("POST /v1/audio/transcriptions (verbose_json) ... ", end="", flush=True)
def t_oai_stt_verbose():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/v1/audio/transcriptions",
                          files={"file": f}, data={"response_format": "verbose_json"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert "segments" in data
    assert "duration" in data
    assert data.get("task") == "transcribe"
    print(f"OK ({len(data['segments'])} segments, {data['duration']:.1f}s)")
test("oai_stt_verbose", t_oai_stt_verbose)

# OpenAI STT — srt
print("POST /v1/audio/transcriptions (srt) ... ", end="", flush=True)
def t_oai_stt_srt():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/v1/audio/transcriptions",
                          files={"file": f}, data={"response_format": "srt"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    assert "-->" in r.text
    print(f"OK ({len(r.text)} chars)")
test("oai_stt_srt", t_oai_stt_srt)

# OpenAI STT — vtt
print("POST /v1/audio/transcriptions (vtt) ... ", end="", flush=True)
def t_oai_stt_vtt():
    with open("/tmp/test_tts.ogg", "rb") as f:
        r = requests.post(f"{HOST}/v1/audio/transcriptions",
                          files={"file": f}, data={"response_format": "vtt"})
    assert r.status_code == 200, f"HTTP {r.status_code}"
    assert r.text.startswith("WEBVTT")
    print(f"OK ({len(r.text)} chars)")
test("oai_stt_vtt", t_oai_stt_vtt)

# Summary
print()
print("=" * 50)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    sys.exit(1)
