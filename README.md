# Voice-Agent

Self-hosted, streaming text-to-speech with zero-shot voice cloning, built to replace ElevenLabs in a real-time voice-agent stack — and to be sold as an API.

- **OpenAI-compatible** `POST /v1/audio/speech` and **ElevenLabs-compatible** `POST /v1/text-to-speech/{voice_id}/stream`: existing clients swap with a base-URL change.
- **Token-level streaming**: first audio in **~150 ms** (English) / **~450 ms** (Hindi and 21 other languages) on an RTX 3090, realtime factor ~0.3.
- **Voice cloning** from a 6-second clip; several clips per person = several emotional registers, chosen per request with `voice`.
- **Telephony formats** out of the box: `pcm`, `wav`, `mulaw`/`alaw` 8 kHz, `mp3`, `opus`, `aac`, `flac`.
- **Sellable**: accounts, hashed API keys, plans with quotas, metering, usage API, and a console (playground, voices, keys, usage, docs, admin).

Built on [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT) — Turbo for English, Multilingual for everything else — with a custom inference path (see *How it is fast*).

---

## Quick start (Windows, NVIDIA GPU)

```powershell
git clone https://github.com/ishan-crd/Voice-Agent.git
cd Voice-Agent
.\setup.ps1                                  # python 3.11 venv, CUDA torch, chatterbox, ffmpeg, node, dashboard build
.\.venv\Scripts\python scripts\smoke_test.py --models turbo   # downloads the model, writes out\smoke.wav
.\start.ps1                                  # http://localhost:8000  (API + console)
```

First start downloads ~4 GB of weights and takes ~80 s to warm both models (CUDA graph capture); later starts take ~25 s.

Linux is the same minus `setup.ps1`: create a Python 3.11 venv, install `torch==2.6.0 torchaudio==2.6.0` from the cu124 index **first**, then `pip install -r requirements.txt`, `apt install ffmpeg`, `cd web && npm i && npm run build`.

### Use it

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")   # no auth until TTS_GATEWAY=1

with client.audio.speech.with_streaming_response.create(
    model="chatterbox", voice="default", input="Hello, am I speaking to Amit?", response_format="pcm",
) as r:
    for chunk in r.iter_bytes(4096):
        play(chunk)            # 16-bit LE mono 24 kHz
```

```bash
# Hindi is auto-detected from the script; ElevenLabs-shaped clients work too
curl -N "http://localhost:8000/v1/text-to-speech/default/stream?output_format=ulaw_8000" \
  -H "xi-api-key: anything" -H "content-type: application/json" \
  -d '{"text":"नमस्ते, क्या मैं अमित से बात कर रहा हूँ?"}' > hindi.ulaw
```

### Clone a voice

Drop `voices/ishan_calm.wav` (6–15 s, one speaker, no music) in the folder and restart, or upload without restarting:

```bash
curl -X POST http://localhost:8000/v1/voices -F name=ishan_calm -F language=en -F file=@ishan_calm.wav
```

The console at `/app/voices/` does the same with a file picker or the browser microphone. Speaker conditionals are cached under `voices/.cache/`, so restarts are instant and a clone is ready on both models in ~0.3 s.

---

## Architecture

```
client (OpenAI SDK / ElevenLabs client / Bolna / Vapi / LiveKit / Pipecat)
   │  POST /v1/audio/speech   or   /v1/text-to-speech/{voice}/stream
   ▼
┌──────────────────────────────── FastAPI, one process ────────────────────────────────┐
│ gateway/   auth (hashed keys) · plans & quotas · concurrency slots · metering (SQLite)│
│ engine/server.py   request schema · voice resolve · language routing · chunker       │
│                    per-chunk streaming jobs → asyncio queues → encoder → HTTP chunks │
│ engine/audio.py    pcm · wav · G.711 µ/A-law · ffmpeg (mp3/opus/aac/flac) · limiter │
│ engine/voices.py   voices/*.wav → Conditionals per model, disk-cached                │
│ engine/models.py   GpuWorker: ONE thread owns the GPU, (priority, FIFO) job queue    │
│ engine/streaming.py  token-level streamers (Turbo, Multilingual) — see below         │
│ web/out            static console served at /                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
   ▼
 RTX 3090: Turbo (fp16 GPT2 T3 + mean-flow S3Gen) · Multilingual (fp16 Llama T3 + CFM S3Gen) ≈ 5 GB
```

**Request path.** Text → sentence chunks (first chunk deliberately short) → each chunk is a GPU job whose blocks are pushed to the event loop as soon as they exist → encoded → yielded. The first chunk of every request has priority over later chunks of every other request, so a new caller hears something before a long monologue finishes. Client disconnect cancels the remaining work.

**Routing.** `language` if given, else Devanagari → `hi`, else the voice's language, else `en`. English goes to Turbo, everything else to Multilingual. `model: "chatterbox-turbo" | "chatterbox-multilingual"` forces it.

### How it is fast

The stock `generate()` runs the whole autoregressive decoder, then the whole vocoder. `engine/streaming.py` replaces both loops:

| Stage | Stock | Here | How |
|---|---|---|---|
| T3 decode step (Turbo) | 15.7 ms/token | **4.4 ms** | fp16, `StaticCache`, one step captured as a CUDA graph |
| T3 decode step (Multilingual, CFG batch 2) | ~45 ms/token | **7.2 ms** | same, plus a text-length runaway cap instead of the attention-hook analyzer |
| Vocoder per block (Turbo, 2-step mean-flow) | 135 ms | **~30 ms** | whole Euler solve captured per 64-frame length bucket |
| Vocoder per block (Multilingual, 10-step CFG) | ~600 ms | **~250 ms** | same, 20 estimator passes in one graph |
| First audio | whole sentence | **12 tokens** | blocks vocoded as tokens arrive: fixed per-request noise, HiFT source cache, crossfade at joins |

Measured on the built-in voice, RTX 3090, over HTTP with keep-alive: English TTFB p50 **147–156 ms**, Hindi **~450 ms**, RTF 0.27–0.54, playback margin ≥ +85 ms (audio always arrives before the previous block finishes playing).

`scripts/profile_*.py` reproduce every number above; `scripts/bench_ttfb.py` measures the HTTP path.

**Lock the GPU clocks.** A voice server is bursty; between requests the driver drops to idle clocks and ramps back per request, which adds 200-400 ms to first audio and makes Whisper 2-3x slower while the TTS engine is resident. `setup.ps1` installs a logon task that runs `nvidia-smi -lgc 1695,1830` (RTX 3090 values; ~40 W extra at idle, undo with `nvidia-smi -rgc`). On Linux: `sudo nvidia-smi -lgc 1695,1830`.

### Talk latency (release the button -> first sound), RTX 3090

| Turn | STT | first LLM token | first audio |
|---|---|---|---|
| spoken English | ~200 ms | ~300 ms | **~490 ms** |
| typed English | - | ~30 ms | ~320 ms |
| typed Hindi | - | ~70 ms | ~870 ms |

How: the browser streams 16 kHz PCM while the button is held (no webm encode/decode on release); Whisper runs the instant the button is released; the first clause of the reply (>= 5 words, or 12 words) goes to TTS before the sentence ends; 100-130 ms of leading and 230-640 ms of trailing silence are trimmed from every TTS chunk.

---

## Selling it: gateway + console

```powershell
# .env
TTS_GATEWAY=1
TTS_ADMIN_KEY=<long random string>
TTS_PUBLIC_URL=https://tts.yourdomain.com

.\.venv\Scripts\python scripts\admin.py create-account "Acme Voice" --email ops@acme.com --plan starter
#  account  acc_...   api key  va_...   (shown once)
```

Give the customer `https://tts.yourdomain.com/app/?key=va_...` — it signs their browser in. Plans (edit `gateway/db.py`):

| Plan | chars / month | concurrent streams | rpm | voices |
|---|---|---|---|---|
| free | 50k | 1 | 30 | 1 |
| starter | 2M (~37 h) | 3 | 120 | 10 |
| scale | 10M (~185 h) | 8 | 600 | 50 |

Every request is metered (chars, audio seconds, time-to-first-audio, status). `GET /v1/usage`, `GET /v1/usage/recent`, `GET /v1/me` for customers; `/admin/*` and the console's Admin page for you. Limits come back as OpenAI-style errors with codes `rate_limit_exceeded`, `insufficient_quota`, `concurrency_limit`.

**Exposing the home box.** `.\start.ps1 -Tunnel` opens a Cloudflare quick tunnel; for a real domain use a named tunnel (`cloudflared tunnel create`) and put Cloudflare's WAF/rate limiting in front. Set `TTS_CORS_ORIGINS` to your console origin once it is not served from the same process.

**Moving to rented GPUs.** The engine and gateway are one process today for simplicity; `gateway/` has no dependency on `engine/`, so the split is: run `engine.server` with `TTS_GATEWAY=0` on each GPU box behind a private network, and put a thin proxy with `gateway/` in front. Postgres replaces SQLite at that point (`gateway/db.py` is plain SQL).

---

## API reference (short)

`POST /v1/audio/speech` — body: `input` (≤4096 chars), `voice`, `response_format` (`pcm|wav|mp3|opus|aac|flac|mulaw|alaw`), `speed` 0.5–2 (pitch preserved), `language`, `sample_rate`, `exaggeration`, `temperature`, `seed`, `stream` (false = buffered). Header `X-Sample-Rate` on the response.

`POST /v1/text-to-speech/{voice_id}[/stream]?output_format=…` — ElevenLabs body (`text`, `model_id`, `voice_settings`, `language_code`); `output_format` like `pcm_16000`, `mp3_44100_128`, `ulaw_8000`.

`GET/POST/DELETE /v1/voices` · `GET /v1/models` · `GET /health` · `GET /v1/stats` · `GET /v1/me` · `GET /v1/usage` · `GET/POST/DELETE /v1/keys` · `/admin/*`

Auth: `Authorization: Bearer va_…` or `xi-api-key: va_…`.

## Configuration

Everything is in `.env.example` (all `TTS_*`). The ones that matter: `TTS_MODELS` (`turbo`, `multilingual` or both), `TTS_GATEWAY`, `TTS_ADMIN_KEY`, `TTS_FIRST_BLOCK_TOKENS` (latency vs. stability), `TTS_REF_SECONDS` (vocoder prompt length), `TTS_GAIN_MULTILINGUAL`.

## Layout

```
engine/      config · models (GPU worker, loaders) · streaming (token-level, CUDA graphs) · chunking · audio · voices · server
gateway/     db (SQLite store) · auth (principal, quotas, concurrency, metering)
web/         Next.js console (static export)
scripts/     smoke_test · bench_ttfb · profile_* · admin
voices/      reference clips (+ .cache/)
```

## Roadmap

- **Block-level round-robin across concurrent streams** (one KV-cache slot per active stream): today the GPU serialises whole sentence chunks, so simultaneous first requests queue behind each other (~1 s per collision).
- **WebSocket endpoint** (ElevenLabs `stream-input` shape) for LLM-token-in / audio-out without sentence buffering.
- Billing (Razorpay + Stripe) on top of the metering table; email sign-up in the console.
- Multi-GPU: N engine processes behind the gateway.
- Fine-tunes for Indian English / Hinglish; paralinguistic tags (`[laugh]`) that Turbo supports.

## Notes on licensing and ethics

Chatterbox code is MIT; check the model cards on Hugging Face for the weights before charging money. Outputs carry Resemble's Perth watermark (keep `TTS_WATERMARK=1`). Only clone voices you own or have permission to use — the console says so on the upload form; put it in your ToS too.
