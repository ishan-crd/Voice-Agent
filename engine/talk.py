"""Live conversation: audio in -> Whisper -> LLM (streamed) -> TTS (streamed) -> audio out.

One WebSocket per conversation.  Messages:

  client -> server   JSON {"type":"config", "voice", "language", "system"}   (optional, any time)
                     JSON {"type":"turn", "audio": <base64>, "mime": "audio/webm"}   one user utterance
                     JSON {"type":"text", "text": "..."}                     typed turn (no STT)
                     JSON {"type":"cancel"}                                  barge-in
                     JSON {"type":"reset"}                                   clear history

  server -> client   JSON {"type":"transcript", "text", "language", "ms"}
                     JSON {"type":"token", "text"}                           LLM tokens as they arrive
                     JSON {"type":"sentence", "text"}                        a sentence was handed to TTS
                     BINARY                                                  16-bit PCM 24 kHz mono
                     JSON {"type":"done", "reply", "timings": {...}}
                     JSON {"type":"error", "message"}

Timings are measured from the moment the turn arrives at the server:
stt_ms, llm_first_token_ms, first_audio_ms, total_ms.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .audio import pcm16_bytes, soft_limit, transcode_to_wav
from .chunking import chunk_text, split_sentences
from .config import settings
from .models import GenParams

log = logging.getLogger("tts.talk")

_TERM = re.compile(r"[.!?।॥。！？]\s*$")


class LLM:
    """Streaming chat over any OpenAI-compatible endpoint (Ollama by default)."""

    def __init__(self) -> None:
        self.base = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.key = settings.llm_api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))

    async def available(self) -> tuple[bool, str]:
        try:
            r = await self.client.get(f"{self.base}/models", headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code >= 400:
                return False, f"{self.base} answered {r.status_code}"
            ids = [m.get("id") for m in r.json().get("data", [])]
            if ids and self.model not in ids and not any(i.startswith(self.model) for i in ids):
                return False, f"model {self.model!r} not found at {self.base} (have: {', '.join(ids[:6])})"
            return True, f"{self.model} @ {self.base}"
        except Exception as e:  # noqa: BLE001
            return False, f"cannot reach {self.base}: {e.__class__.__name__}"

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": 0.6, "max_tokens": 200}
        async with self.client.stream("POST", f"{self.base}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.key}"}) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"LLM {r.status_code}: {(await r.aread())[:200].decode(errors='ignore')}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except Exception:  # noqa: BLE001
                    continue
                if delta:
                    yield delta


def _decode_turn(b64: str, mime: str) -> np.ndarray:
    """base64 browser audio -> float32 mono 16 kHz."""
    import librosa
    import soundfile as sf

    suffix = ".webm" if "webm" in mime else ".ogg" if "ogg" in mime else ".wav" if "wav" in mime else ".m4a" if ("mp4" in mime or "m4a" in mime) else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(base64.b64decode(b64))
        raw = Path(f.name)
    wav = raw.with_suffix(".stt.wav")
    try:
        transcode_to_wav(raw, wav, sr=16_000)
        x, sr = sf.read(wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16_000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16_000)
        return x
    finally:
        raw.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


class Conversation:
    def __init__(self, ws: WebSocket, engine, registry, stt, llm: LLM) -> None:
        self.ws = ws
        self.engine = engine
        self.registry = registry
        self.stt = stt
        self.llm = llm
        self.voice = settings.default_voice
        self.language: str | None = None
        self.system = settings.llm_system_prompt
        self.history: list[dict] = []
        self.cancel = threading.Event()
        self.turn_task: asyncio.Task | None = None
        self.meter = None  # set by the server: (chars, audio_s, ttfa_ms, total_ms, lang, kind) -> None
        self._audio_s = 0.0
        self._kind = ""

    async def send(self, **msg) -> None:
        await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    # ----------------------------------------------------------------- loop
    async def run(self) -> None:
        ok, info = await self.llm.available()
        await self.send(type="ready", llm=info, llm_ok=ok, stt=self.stt is not None and self.stt.model_id, voice=self.voice)
        try:
            while True:
                msg = json.loads(await self.ws.receive_text())
                t = msg.get("type")
                if t == "config":
                    self.voice = msg.get("voice") or self.voice
                    self.language = msg.get("language") or None
                    if msg.get("system"):
                        self.system = msg["system"]
                elif t == "reset":
                    self.history.clear()
                    await self.send(type="reset")
                elif t == "cancel":
                    await self._cancel()
                elif t in ("turn", "text"):
                    await self._cancel()
                    self.cancel = threading.Event()
                    self.turn_task = asyncio.create_task(self._turn(msg))
        except WebSocketDisconnect:
            await self._cancel()

    async def _cancel(self) -> None:
        self.cancel.set()
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ----------------------------------------------------------------- turn
    async def _turn(self, msg: dict) -> None:
        t0 = time.perf_counter()
        timings: dict[str, float | None] = {"stt_ms": None, "llm_first_token_ms": None, "first_audio_ms": None, "total_ms": None}
        cancel = self.cancel
        try:
            # 1. hear
            if msg["type"] == "turn":
                if self.stt is None:
                    await self.send(type="error", message="speech-to-text is disabled on this server (TTS_STT=0)")
                    return
                audio = await asyncio.to_thread(_decode_turn, msg["audio"], msg.get("mime", "audio/webm"))
                if len(audio) < 16_000 * 0.3:
                    await self.send(type="error", message="that was too short - hold the button while you speak")
                    return
                tr = await asyncio.to_thread(self.stt.transcribe, audio, self.language)
                timings["stt_ms"] = round((time.perf_counter() - t0) * 1000)
                user_text, lang = tr.text, tr.language
                await self.send(type="transcript", text=user_text, language=lang, ms=timings["stt_ms"])
                if not user_text.strip():
                    await self.send(type="error", message="I didn't catch any words in that")
                    return
            else:
                user_text = msg["text"].strip()
                lang = self.language or ("hi" if re.search(r"[ऀ-ॿ]", user_text) else "en")
                timings["stt_ms"] = 0
                await self.send(type="transcript", text=user_text, language=lang, ms=0)

            # 2. think + 3. speak, overlapped: sentences go to TTS as soon as they complete
            messages = [{"role": "system", "content": self.system}, *self.history[-12:], {"role": "user", "content": user_text}]
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            speaker = asyncio.create_task(self._speak(sentences, lang, t0, timings, cancel))

            reply = ""
            buf = ""
            first = True
            async for tok in self.llm.stream(messages):
                if cancel.is_set():
                    break
                if first:
                    first = False
                    timings["llm_first_token_ms"] = round((time.perf_counter() - t0) * 1000)
                reply += tok
                buf += tok
                await self.send(type="token", text=tok)
                if _TERM.search(buf) and len(buf.strip()) > 2:
                    parts = split_sentences(buf)
                    for s in parts:
                        await sentences.put(s)
                    buf = ""
            if buf.strip() and not cancel.is_set():
                await sentences.put(buf.strip())
            await sentences.put(None)
            await speaker
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
            if not cancel.is_set():
                self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply.strip()}]
                await self.send(type="done", reply=reply.strip(), timings=timings)
            if self.meter:
                try:
                    self.meter(len(reply), round(self._audio_s, 2), timings["first_audio_ms"], timings["total_ms"], lang, self._kind)
                except Exception:  # noqa: BLE001
                    log.exception("talk metering failed")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            await self.send(type="error", message=str(e))

    async def _speak(self, sentences: asyncio.Queue, lang: str, t0: float, timings: dict, cancel: threading.Event) -> None:
        try:
            voice = self.registry.resolve(self.voice)
        except KeyError:
            voice = self.registry.resolve(None)
        kind = self.engine.pick(lang)
        self._kind = kind
        self._audio_s = 0.0
        conds = await self.registry.ensure(voice, kind)
        gain = settings.gain_turbo if kind == "turbo" else settings.gain_multilingual
        params = GenParams()
        while True:
            s = await sentences.get()
            if s is None or cancel.is_set():
                return
            await self.send(type="sentence", text=s)
            for chunk in chunk_text(s):
                q = self.engine.worker.submit_stream(
                    lambda c=chunk: self.engine.stream(kind, c, conds, lang, params), priority=0, cancel=cancel
                )
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if cancel.is_set():
                        break
                    if timings["first_audio_ms"] is None:
                        timings["first_audio_ms"] = round((time.perf_counter() - t0) * 1000)
                        await self.send(type="audio_start", ms=timings["first_audio_ms"])
                    self._audio_s += len(item) / self.engine.sr
                    await self.ws.send_bytes(pcm16_bytes(soft_limit(item, gain)))
