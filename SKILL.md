# Clone Voice Service

Voice cloning TTS and speech-to-text in a single API. Generate speech in cloned voices, transcribe audio to text.

## Endpoints

### Text-to-Speech

```
POST http://spark-01:3030/tts
Content-Type: application/json
```

```json
{
  "text": "The text you want spoken.",
  "voice": "paul",
  "quality": "fast",
  "format": "ogg"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to speak |
| `voice` | string | *required* | Voice profile name |
| `quality` | string | `"quality"` | `"fast"` (16 steps) or `"quality"` (32 steps) |
| `format` | string | `"ogg"` | `"ogg"` (Opus), `"flac"`, or `"mp3"` |
| `speed` | float | `1.0` | Playback speed (0.1-3.0) |
| `seed` | int | `-1` | RNG seed (-1 for random) |
| `cfg_strength` | float | `2.0` | Guidance strength (0.1-5.0) |

Returns audio binary data.

### Speech-to-Text

```
POST http://spark-01:3030/stt
Content-Type: multipart/form-data
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | file | *required* | Audio file (any format: ogg, wav, mp3, flac, webm, etc.) |
| `language` | string | auto-detect | Language code: `"en"`, `"fr"`, `"de"`, etc. |
| `format` | string | `"json"` | `"json"`, `"text"` (plain text), or `"verbose"` (with timestamps) |

Returns transcribed text.

### List Voices

```
GET http://spark-01:3030/voices
```

Fetch available voices before generating audio. Returns JSON keyed by voice name.

### OpenAI-Compatible STT

```
POST http://spark-01:3030/v1/audio/transcriptions
Content-Type: multipart/form-data
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *required* | Audio file (any format) |
| `model` | string | `"whisper-1"` | Accepted but ignored (uses configured backend) |
| `language` | string | auto-detect | Language code: `"en"`, `"fr"`, `"de"`, etc. |
| `prompt` | string | none | Optional context to improve accuracy |
| `response_format` | string | `"json"` | `"json"`, `"text"`, `"verbose_json"`, `"srt"`, `"vtt"` |
| `temperature` | float | none | Sampling temperature |

Compatible with any OpenAI STT client — set `baseUrl` to `http://spark-01:3030/v1`.

### OpenAI-Compatible TTS

```
POST http://spark-01:3030/v1/audio/speech
Content-Type: application/json
```

```json
{
  "model": "tts-1",
  "input": "The text you want spoken.",
  "voice": "paul",
  "response_format": "opus",
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"tts-1"` | Accepted but ignored (always uses F5-TTS) |
| `input` | string | *required* | Text to speak |
| `voice` | string | `"paul"` | Voice profile name |
| `response_format` | string | `"opus"` | `"opus"`, `"mp3"`, `"flac"`, `"wav"`, `"aac"`, `"pcm"` |
| `speed` | float | `1.0` | Playback speed |

Returns audio binary data. Compatible with any OpenAI TTS client — set `baseUrl` to `http://spark-01:3030/v1`.

### Health Check

```
GET http://spark-01:3030/health
```

Returns status, device (cuda/cpu), and which models are loaded.

## Examples

**TTS with curl:**

```bash
curl -s -X POST http://spark-01:3030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "...", "voice": "anomaly", "quality": "fast", "format": "ogg"}' \
  -o /tmp/tts_output.ogg
```

**TTS with Python:**

```python
requests.post(
    'http://spark-01:3030/tts',
    json={'text': '...', 'voice': 'anomaly', 'quality': 'fast', 'format': 'ogg'}
)
```

**STT with curl:**

```bash
curl -s -X POST http://spark-01:3030/stt \
  -F "audio=@recording.ogg" \
  -F "format=text"
```

**STT with Python:**

```python
with open('recording.ogg', 'rb') as f:
    requests.post(
        'http://spark-01:3030/stt',
        files={'audio': f},
        data={'format': 'json'}
    )
```

## Delivering Audio

Send generated audio using the messaging platform's **voice/audio message** feature — not as a file attachment. Default OGG Opus format is compatible with Telegram and WhatsApp.

## Adding Voices

Drop a `.wav` and `.txt` pair into the `voices/` directory. No restart needed — voices are discovered automatically.

## Notes

- First request loads the model (~30s). Subsequent requests are sub-second.
- TTS model uses ~1.2GB GPU memory. STT model varies by size (tiny: 39M, turbo: 809M).
- Runs on GPU (CUDA) or CPU. CPU is slower but works on any system.
- The `paul` voice should be used when speaking as or on behalf of Paul.
