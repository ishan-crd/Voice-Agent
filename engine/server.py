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
import os
import re
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gateway import auth
from gateway.db import PLANS, Principal

from .audio import FORMATS, StreamEncoder, transcode_to_wav
from .chunking import chunk_text
from .config import settings
from .models import GenParams, TTSEngine
from .talk import LLM, Conversation
from .voices import VoiceRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts.server")

engine = TTSEngine()
registry = VoiceRegistry(engine)
stt = None  # Whisper, loaded in lifespan when TTS_STT=1
llm = LLM()

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
    global stt
    engine.load()
    registry.load()
    if settings.stt:
        try:
            from .stt import Whisper

            stt = Whisper(settings.stt_model, settings.device)
        except Exception:  # noqa: BLE001
            log.exception("could not load Whisper; the Talk page will be disabled")
    ok, info = await llm.available()
    log.info("llm: %s (%s)", info, "ok" if ok else "unavailable - Talk page will report this")
    if ok:
        await llm.warm()
    log.info("ready on http://%s:%d  models=%s", settings.host, settings.port, list(engine.models))
    yield


app = FastAPI(title="Voice-Agent TTS", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate"],
)


# --------------------------------------------------------------------- schema
class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech body plus Chatterbox extensions."""

    model: str = "chatterbox"
    input: str = Field(min_length=1, max_length=4096)
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
    text: str = Field(min_length=1, max_length=4096)
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


def _can_use_voice(p: Principal, voice_id: str) -> bool:
    if p.is_admin or auth.store is None:
        return True
    v = registry.voices.get(voice_id)
    if v is not None and v.builtin:
        return True
    return auth.store.voice_owner(voice_id) == p.account_id


async def synthesize_stream(
    *,
    p: Principal,
    endpoint: str,
    text: str,
    voice_id: str,
    fmt: str,
    speed: float = 1.0,
    sample_rate: int | None = None,
    language: str | None = None,
    model: str | None = None,
    params: GenParams,
) -> tuple[StreamEncoder, AsyncIterator[bytes]]:
    auth.check_quota(p, len(text))
    try:
        voice = registry.resolve(voice_id)
    except KeyError:
        raise HTTPException(404, f"unknown voice {voice_id!r}; GET /v1/voices for the list")
    if not _can_use_voice(p, voice.id):
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

    gain = settings.gain_turbo if kind == "turbo" else settings.gain_multilingual
    try:
        enc = StreamEncoder(fmt, engine.sr, sample_rate=sample_rate, gain=gain)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    cancel = threading.Event()
    t_start = time.perf_counter()
    stats["requests"] += 1
    stats["chars"] += len(text)
    slot = auth.concurrency_slot(p)
    slot.__enter__()  # raises 429 before any audio if the account is at its limit

    async def gen() -> AsyncIterator[bytes]:
        queues: deque[asyncio.Queue] = deque()
        next_idx = 0
        first = True
        audio_secs = 0.0
        ttfa_ms: float | None = None
        status = 200

        def submit(i: int) -> None:
            # first chunk of any request outranks later chunks of every other request:
            # a new caller hears something quickly, long monologues fill in behind
            queues.append(
                engine.worker.submit_stream(
                    lambda: engine.stream(kind, chunks[i], conds, lang, params, speed=speed),
                    priority=0 if i == 0 else 1,
                    cancel=cancel,
                )
            )

        try:
            while next_idx < min(settings.lookahead, len(chunks)):
                submit(next_idx)
                next_idx += 1
            while queues:
                q = queues.popleft()
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    audio_secs += len(item) / engine.sr
                    for b in enc.encode(item):
                        if first:
                            first = False
                            ttfa_ms = (time.perf_counter() - t_start) * 1000
                            _ttfa_window.append(ttfa_ms)
                            stats["ttfa_ms_last"] = round(ttfa_ms, 1)
                            stats["ttfa_ms_avg"] = round(sum(_ttfa_window) / len(_ttfa_window), 1)
                        yield b
                if next_idx < len(chunks):
                    submit(next_idx)
                    next_idx += 1
            for b in enc.finish():
                yield b
            stats["audio_seconds"] += audio_secs
            log.info(
                "voice=%s model=%s lang=%s fmt=%s chunks=%d chars=%d audio=%.1fs ttfa=%sms total=%.0fms",
                voice.id, kind, lang, fmt, len(chunks), len(text), audio_secs,
                stats["ttfa_ms_last"], (time.perf_counter() - t_start) * 1000,
            )
        except (asyncio.CancelledError, GeneratorExit):
            status = 499
            log.info("client disconnected; cancelling %d pending chunks", len(chunks) - next_idx + len(queues))
            raise
        except Exception:
            status = 500
            stats["errors"] += 1
            raise
        finally:
            cancel.set()
            enc.close()
            slot.__exit__(None, None, None)
            auth.record(
                p, endpoint=endpoint, model=kind, voice=voice.id, language=lang, format=fmt,
                chars=len(text), audio_s=round(audio_secs, 2), ttfa_ms=ttfa_ms,
                total_ms=round((time.perf_counter() - t_start) * 1000, 1), status=status,
            )

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
@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest, p: Principal = Depends(auth.principal)):
    params = GenParams(
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
        seed=req.seed,
    )
    enc, body = await synthesize_stream(
        p=p,
        endpoint="openai.speech",
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


async def _eleven(p: Principal, voice_id: str, req: ElevenRequest, output_format: str | None, stream: bool) -> Response:
    fmt, sr = _eleven_output(output_format)
    vs = req.voice_settings or ElevenVoiceSettings()
    exaggeration = 0.5
    if vs.style is not None:
        exaggeration = max(0.0, min(2.0, vs.style))
    elif vs.stability is not None:
        exaggeration = max(0.0, min(1.0, 1.0 - vs.stability))
    params = GenParams(exaggeration=exaggeration, seed=req.seed)
    enc, body = await synthesize_stream(
        p=p,
        endpoint="eleven.tts",
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


@app.post("/v1/text-to-speech/{voice_id}/stream")
async def eleven_stream(voice_id: str, req: ElevenRequest, output_format: str | None = None, p: Principal = Depends(auth.principal)):
    return await _eleven(p, voice_id, req, output_format, stream=True)


@app.post("/v1/text-to-speech/{voice_id}")
async def eleven_full(voice_id: str, req: ElevenRequest, output_format: str | None = None, p: Principal = Depends(auth.principal)):
    return await _eleven(p, voice_id, req, output_format, stream=False)


# ----------------------------------------------------------------- voices
@app.get("/v1/voices")
async def list_voices(p: Principal = Depends(auth.principal)):
    voices = [v for v in registry.list() if _can_use_voice(p, v["voice_id"])]
    return {"voices": voices}


@app.post("/v1/voices", status_code=201)
async def add_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    description: str = Form(""),
    language: str = Form("en"),
    exaggeration: float = Form(0.5),
    overwrite: bool = Form(False),
    mode: str = Form("quick"),  # quick | pro (long recording: best window + averaged identity)
    p: Principal = Depends(auth.principal),
):
    if auth.store is not None and not p.is_admin:
        owner = auth.store.voice_owner(name)
        if owner is not None and owner != p.account_id:
            raise HTTPException(409, f"voice name {name!r} is taken")
        if p.max_voices and owner is None and len(auth.store.account_voices(p.account_id)) >= p.max_voices:
            raise HTTPException(429, f"voice limit: {p.max_voices} on the {p.plan} plan")
    suffix = Path(file.filename or "clip.wav").suffix.lower() or ".bin"
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "recording too large (200 MB max)")
    fd, raw_name = tempfile.mkstemp(suffix=suffix, dir=registry.cache_dir)
    os.close(fd)  # Windows locks the file while the descriptor is open
    raw = Path(raw_name)
    raw.write_bytes(data)
    # normalise whatever the client sent (browser webm/opus, m4a, mp3, ...) to a
    # clean 24 kHz mono wav so the cloner never has to decode it itself
    tmp = raw.with_suffix(".wav") if suffix != ".wav" else raw
    if tmp != raw:
        try:
            transcode_to_wav(raw, tmp)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        finally:
            raw.unlink(missing_ok=True)
    try:
        voice = await registry.add(
            name, tmp, description=description, language=language, exaggeration=exaggeration, overwrite=overwrite, mode=mode
        )
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(409, f"voice {name!r} exists; pass overwrite=true")
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except AssertionError as e:  # chatterbox: "Audio prompt must be longer than 5 seconds!"
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e) or "reference clip too short (need > 5 s)")
    if auth.store is not None and not p.is_admin:
        auth.store.claim_voice(voice.id, p.account_id)
    return voice.to_public()


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str, p: Principal = Depends(auth.principal)):
    if auth.store is not None and not p.is_admin and auth.store.voice_owner(voice_id) != p.account_id:
        raise HTTPException(404, "no such voice")
    try:
        registry.delete(voice_id)
    except KeyError:
        raise HTTPException(404, "no such voice")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if auth.store is not None:
        auth.store.release_voice(voice_id)
    return {"deleted": voice_id}


# --------------------------------------------------------- account (self)
def _need_store():
    if auth.store is None:
        raise HTTPException(404, "gateway disabled (TTS_GATEWAY=0)")
    return auth.store


@app.get("/v1/me")
async def me(p: Principal = Depends(auth.principal)):
    if p.is_admin:
        return {"account_id": p.account_id, "name": p.account_name, "plan": p.plan, "admin": True}
    st = _need_store()
    acc = st.get_account(p.account_id)
    return {
        "account_id": acc["id"], "name": acc["name"], "email": acc["email"], "plan": acc["plan"],
        "limits": {"chars_per_month": acc["char_limit"], "concurrency": acc["max_concurrency"], "rpm": acc["rpm"], "voices": acc["max_voices"]},
        "month_chars": st.month_chars(acc["id"]),
        "admin": False,
    }


@app.get("/v1/usage")
async def usage(days: int = 30, p: Principal = Depends(auth.principal)):
    st = _need_store()
    return st.usage_summary(None if p.is_admin else p.account_id, days=max(1, min(days, 365)))


@app.get("/v1/usage/recent")
async def usage_recent(limit: int = 50, p: Principal = Depends(auth.principal)):
    st = _need_store()
    return {"rows": st.recent(None if p.is_admin else p.account_id, limit=max(1, min(limit, 500)))}


@app.get("/v1/keys")
async def list_keys(p: Principal = Depends(auth.principal)):
    st = _need_store()
    if p.is_admin:
        raise HTTPException(400, "admin: use /admin/accounts/{id}/keys")
    return {"keys": st.list_keys(p.account_id)}


class KeyCreate(BaseModel):
    name: str = "default"


@app.post("/v1/keys", status_code=201)
async def create_key(body: KeyCreate, p: Principal = Depends(auth.principal)):
    st = _need_store()
    if p.is_admin:
        raise HTTPException(400, "admin: use /admin/accounts/{id}/keys")
    raw, rec = st.create_key(p.account_id, body.name)
    return {**rec, "key": raw}


@app.delete("/v1/keys/{key_id}")
async def revoke_key(key_id: str, p: Principal = Depends(auth.principal)):
    st = _need_store()
    st.revoke_key(key_id, None if p.is_admin else p.account_id)
    return {"revoked": key_id}


# ---------------------------------------------------------------- admin
class AccountCreate(BaseModel):
    name: str
    email: str | None = None
    plan: str = "free"


@app.get("/admin/accounts", dependencies=[Depends(auth.admin)])
async def admin_accounts():
    st = _need_store()
    out = []
    for a in st.list_accounts():
        out.append({**a, "month_chars": st.month_chars(a["id"]), "keys": st.list_keys(a["id"]), "voices": sorted(st.account_voices(a["id"]))})
    return {"accounts": out, "plans": PLANS}


@app.post("/admin/accounts", status_code=201, dependencies=[Depends(auth.admin)])
async def admin_create_account(body: AccountCreate):
    st = _need_store()
    try:
        acc = st.create_account(body.name, body.email, body.plan)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 - unique email
        raise HTTPException(409, f"could not create account: {e}")
    raw, rec = st.create_key(acc["id"], "default")
    return {"account": acc, "key": {**rec, "key": raw}}


class PlanSet(BaseModel):
    plan: str


@app.put("/admin/accounts/{account_id}/plan", dependencies=[Depends(auth.admin)])
async def admin_set_plan(account_id: str, body: PlanSet):
    st = _need_store()
    if body.plan not in PLANS:
        raise HTTPException(400, f"unknown plan; choose from {sorted(PLANS)}")
    return st.set_plan(account_id, body.plan)


@app.post("/admin/accounts/{account_id}/keys", status_code=201, dependencies=[Depends(auth.admin)])
async def admin_create_key(account_id: str, body: KeyCreate):
    st = _need_store()
    st.get_account(account_id)
    raw, rec = st.create_key(account_id, body.name)
    return {**rec, "key": raw}


@app.get("/admin/usage", dependencies=[Depends(auth.admin)])
async def admin_usage(days: int = 30, account_id: str | None = None):
    st = _need_store()
    return st.usage_summary(account_id, days=max(1, min(days, 365)))


# ------------------------------------------------------------------- talk
@app.websocket("/v1/talk")
async def talk(ws: WebSocket):
    # browsers cannot set headers on a WebSocket: the key comes as ?api_key=
    try:
        p = await auth.principal(ws)  # type: ignore[arg-type]  (reads headers/query the same way)
    except HTTPException as e:
        await ws.close(code=4401, reason=str(e.detail)[:120])
        return
    await ws.accept()
    conv = Conversation(ws, engine, registry, stt, llm)

    def meter(chars: int, audio_s: float, ttfa_ms: float | None, total_ms: float | None, lang: str, kind: str) -> None:
        auth.record(p, endpoint="talk", model=kind, voice=conv.voice, language=lang, format="pcm", chars=chars, audio_s=audio_s, ttfa_ms=ttfa_ms, total_ms=total_ms, status=200)

    conv.meter = meter
    await conv.run()


@app.get("/v1/talk/status")
async def talk_status(p: Principal = Depends(auth.principal)):
    ok, info = await llm.available()
    return {"stt": getattr(stt, "model_id", None), "llm": info, "llm_ok": ok}


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
        "stt": getattr(stt, "model_id", None),
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


# ------------------------------------------------------------ dashboard
# `cd web && npm run build` writes the static export to web/out; when it
# exists the console is served from this same process at / and /app.
_WEB = Path(__file__).resolve().parents[1] / "web" / "out"
if _WEB.exists():
    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")
    log.info("serving dashboard from %s", _WEB)


def main() -> None:
    import uvicorn

    uvicorn.run("engine.server:app", host=settings.host, port=settings.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
