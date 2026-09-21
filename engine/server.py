"""FastAPI app: OpenAI- and ElevenLabs-compatible streaming TTS.

    POST /v1/audio/speech                      OpenAI shape
    POST /v1/text-to-speech/{voice_id}/stream  ElevenLabs shape
    GET  /v1/voices   POST /v1/voices   DELETE /v1/voices/{id}
    GET  /v1/models   GET /health   GET /v1/stats

Run:  uvicorn engine.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .audio import FORMATS, StreamEncoder
from .chunking import chunk_text
from .config import settings
from .models import GenParams, TTSEngine
from .voices import VoiceRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts.server")

engine = TTSEngine()
registry = VoiceRegistry(engine)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_MODEL_ALIASES = {
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
}

stats = {
    "requests": 0,
    "errors": 0,
    "chars": 0,
    "audio_seconds": 0.0,
    "ttfa_ms_last": None,
    "ttfa_ms_avg": None,
    "started_at": time.time(),
}
_ttfa_window: deque[float] = deque(maxlen=200)


# ------------------------------------------------------------------ lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    registry.load()
    log.info("ready on http://%s:%d  models=%s", settings.host, settings.port, list(engine.models))
    yield


app = FastAPI(title="Voice-Agent TTS", version="0.1.0", lifespan=lifespan)


# ----------------------------------------------------------------------- auth
async def require_key(request: Request) -> None:
    keys = settings.api_key_set
    if not keys:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("xi-api-key", "")
    if token not in keys:
        raise HTTPException(401, "invalid or missing API key")


# --------------------------------------------------------------------- schema
class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech body plus Chatterbox extensions."""

    model: str = "chatterbox"
    input: str = Field(min_length=1, max_length=8192)
    voice: str = settings.default_voice
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "mulaw", "alaw"] = "mp3"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    # extensions
    language: str | None = None  # ISO-639-1; auto-detects Devanagari -> hi
    sample_rate: int | None = Field(None, ge=8000, le=48000)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    seed: int | None = None
    stream: bool = True


class ElevenVoiceSettings(BaseModel):
    stability: float | None = None  # mapped to (1 - exaggeration)
    similarity_boost: float | None = None
    style: float | None = None  # mapped to exaggeration
    speed: float | None = None


class ElevenRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8192)
    model_id: str | None = None
    language_code: str | None = None
    voice_settings: ElevenVoiceSettings | None = None
    seed: int | None = None


# ------------------------------------------------------------------ synthesis
def _detect_language(text: str, explicit: str | None, voice_lang: str) -> str:
    if explicit:
        return explicit.lower()
    if _DEVANAGARI.search(text):
        return "hi"
    return voice_lang or "en"


async def synthesize_stream(
    *,
    text: str,
    voice_id: str,
    fmt: str,
    speed: float = 1.0,
    sample_rate: int | None = None,
    language: str | None = None,
    model: str | None = None,
    params: GenParams,
) -> tuple[StreamEncoder, AsyncIterator[bytes]]:
    try:
        voice = registry.resolve(voice_id)
    except KeyError:
        raise HTTPException(404, f"unknown voice {voice_id!r}; GET /v1/voices for the list")

    lang = _detect_language(text, language, voice.language)
    try:
        kind = engine.pick(lang, _MODEL_ALIASES.get((model or "").lower()))
    except ValueError as e:
        raise HTTPException(400, str(e))

    conds = await registry.ensure(voice, kind)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "input has no speakable text")

    try:
        enc = StreamEncoder(fmt, engine.sr, sample_rate=sample_rate, speed=speed)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    cancel = threading.Event()
    t_start = time.perf_counter()
    stats["requests"] += 1
    stats["chars"] += len(text)

    async def gen() -> AsyncIterator[bytes]:
        futs: deque[asyncio.Future] = deque()
        next_idx = 0
        first = True
        audio_secs = 0.0

        def submit(i: int) -> None:
            # first chunk of any request outranks later chunks of every other request:
            # a new caller hears something quickly, long monologues fill in behind
            futs.append(
                engine.worker.submit(
                    lambda: engine.synthesize(kind, chunks[i], conds, lang, params),
                    priority=0.0 if i == 0 else 1.0,
                    cancel=cancel,
                )
            )

        try:
            while next_idx < min(settings.lookahead, len(chunks)):
                submit(next_idx)
                next_idx += 1
            while futs:
                audio = await futs.popleft()
                if next_idx < len(chunks):
                    submit(next_idx)
                    next_idx += 1
                audio_secs += len(audio) / engine.sr
                for b in enc.encode(audio):
                    if first:
                        first = False
                        ttfa = (time.perf_counter() - t_start) * 1000
                        _ttfa_window.append(ttfa)
                        stats["ttfa_ms_last"] = round(ttfa, 1)
                        stats["ttfa_ms_avg"] = round(sum(_ttfa_window) / len(_ttfa_window), 1)
                    yield b
            for b in enc.finish():
                yield b
            stats["audio_seconds"] += audio_secs
            log.info(
                "voice=%s model=%s lang=%s fmt=%s chunks=%d chars=%d audio=%.1fs ttfa=%sms total=%.0fms",
                voice.id, kind, lang, fmt, len(chunks), len(text), audio_secs,
                stats["ttfa_ms_last"], (time.perf_counter() - t_start) * 1000,
            )
        except (asyncio.CancelledError, GeneratorExit):
            log.info("client disconnected; cancelling %d queued chunks", len(chunks) - next_idx + len(futs))
            raise
        except Exception:
            stats["errors"] += 1
            raise
        finally:
            cancel.set()
            for f in futs:
                f.cancel()
            enc.close()

    return enc, gen()


async def _respond(enc: StreamEncoder, body: AsyncIterator[bytes], stream: bool) -> Response:
    headers = {"Cache-Control": "no-store", "X-Sample-Rate": str(enc.sr_out)}
    if stream:
        return StreamingResponse(body, media_type=enc.content_type, headers=headers)
    buf = bytearray()
    async for b in body:
        buf.extend(b)
    return Response(bytes(buf), media_type=enc.content_type, headers=headers)


# ------------------------------------------------------------- OpenAI route
@app.post("/v1/audio/speech", dependencies=[Depends(require_key)])
async def openai_speech(req: SpeechRequest):
    params = GenParams(
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
        seed=req.seed,
    )
    enc, body = await synthesize_stream(
        text=req.input,
        voice_id=req.voice,
        fmt=req.response_format,
        speed=req.speed,
        sample_rate=req.sample_rate,
        language=req.language,
        model=req.model,
        params=params,
    )
    return await _respond(enc, body, req.stream)


# --------------------------------------------------------- ElevenLabs routes
_ELEVEN_FMT = re.compile(r"^(mp3|pcm|ulaw|alaw|opus)_(\d+)(?:_(\d+))?$")


def _eleven_output(output_format: str | None) -> tuple[str, int | None]:
    """'mp3_44100_128' -> ('mp3', 44100); 'pcm_16000' -> ('pcm', 16000); 'ulaw_8000' -> ('mulaw', 8000)."""
    if not output_format:
        return "mp3", None
    m = _ELEVEN_FMT.match(output_format)
    if not m:
        raise HTTPException(400, f"unsupported output_format {output_format!r}")
    codec, sr = m.group(1), int(m.group(2))
    return {"ulaw": "mulaw"}.get(codec, codec), sr


async def _eleven(voice_id: str, req: ElevenRequest, output_format: str | None, stream: bool) -> Response:
    fmt, sr = _eleven_output(output_format)
    vs = req.voice_settings or ElevenVoiceSettings()
    exaggeration = 0.5
    if vs.style is not None:
        exaggeration = max(0.0, min(2.0, vs.style))
    elif vs.stability is not None:
        exaggeration = max(0.0, min(1.0, 1.0 - vs.stability))
    params = GenParams(exaggeration=exaggeration, seed=req.seed)
    enc, body = await synthesize_stream(
        text=req.text,
        voice_id=voice_id,
        fmt=fmt,
        speed=vs.speed or 1.0,
        sample_rate=sr,
        language=req.language_code,
        model=req.model_id,
        params=params,
    )
    return await _respond(enc, body, stream)


@app.post("/v1/text-to-speech/{voice_id}/stream", dependencies=[Depends(require_key)])
async def eleven_stream(voice_id: str, req: ElevenRequest, output_format: str | None = None):
    return await _eleven(voice_id, req, output_format, stream=True)


@app.post("/v1/text-to-speech/{voice_id}", dependencies=[Depends(require_key)])
async def eleven_full(voice_id: str, req: ElevenRequest, output_format: str | None = None):
    return await _eleven(voice_id, req, output_format, stream=False)


# ----------------------------------------------------------------- voices
@app.get("/v1/voices", dependencies=[Depends(require_key)])
async def list_voices():
    return {"voices": registry.list()}


@app.post("/v1/voices", dependencies=[Depends(require_key)], status_code=201)
async def add_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    description: str = Form(""),
    language: str = Form("en"),
    exaggeration: float = Form(0.5),
    overwrite: bool = Form(False),
):
    suffix = Path(file.filename or "clip.wav").suffix.lower() or ".wav"
    if suffix not in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        raise HTTPException(400, "upload wav, flac, mp3, ogg or m4a")
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "clip too large (25 MB max)")
    tmp = Path(tempfile.mkstemp(suffix=suffix, dir=registry.root)[1])
    tmp.write_bytes(data)
    try:
        voice = await registry.add(
            name, tmp, description=description, language=language, exaggeration=exaggeration, overwrite=overwrite
        )
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(409, f"voice {name!r} exists; pass overwrite=true")
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    return voice.to_public()


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_key)])
async def delete_voice(voice_id: str):
    try:
        registry.delete(voice_id)
    except KeyError:
        raise HTTPException(404, "no such voice")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"deleted": voice_id}


# ------------------------------------------------------------------- misc
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    ids = ["tts-1", "tts-1-hd", "chatterbox"] + [f"chatterbox-{k}" for k in engine.models]
    return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "local"} for i in ids]}


@app.get("/health")
async def health():
    gpu = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "vram_used_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            }
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok" if engine.models else "loading",
        "models": list(engine.models),
        "voices": sorted(registry.voices),
        "queue_depth": engine.worker.queue_depth,
        "formats": sorted(FORMATS),
        "gpu": gpu,
    }


@app.get("/v1/stats")
async def get_stats():
    return {**stats, "uptime_s": int(time.time() - stats["started_at"]), "queue_depth": engine.worker.queue_depth}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"error": {"message": str(exc), "type": "server_error"}}, status_code=500)


def main() -> None:
    import uvicorn

    uvicorn.run("engine.server:app", host=settings.host, port=settings.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
