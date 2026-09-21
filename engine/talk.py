"""Live conversation: audio in -> Whisper -> LLM (streamed) -> TTS (streamed) -> audio out.

One WebSocket per conversation.  Messages:

  client -> server   JSON {"type":"config", "voice", "language", "system", "speed"}   (optional, any time)
                     JSON {"type":"turn_start"} · BINARY int16 16 kHz mono frames · JSON {"type":"turn_end"}
                                                                             live mic audio; STT runs the instant turn_end arrives
                     JSON {"type":"turn", "audio": <base64>, "mime": "audio/webm"}   one whole utterance (API clients)
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
_CLAUSE = re.compile(r"[,;:—–]\s*$")
FIRST_CLAUSE_MIN_WORDS = 5  # hand the first clause to TTS early instead of waiting for the full sentence
FIRST_CLAUSE_MAX_WORDS = 12

# script -> language the TTS should use for a sentence; Latin falls back to the
# conversation language so English-only Turbo never receives text it can't speak
_SCRIPTS = [
    (re.compile(r"[\u0900-\u097F]"), "hi"),
    (re.compile(r"[\u0600-\u06FF]"), "ar"),
    (re.compile(r"[\u0400-\u04FF]"), "ru"),
    (re.compile(r"[\u3040-\u30FF]"), "ja"),
    (re.compile(r"[\uAC00-\uD7AF]"), "ko"),
    (re.compile(r"[\u4E00-\u9FFF]"), "zh"),
    (re.compile(r"[\u0370-\u03FF]"), "el"),
    (re.compile(r"[\u0590-\u05FF]"), "he"),
]
# every non-Latin letter range we know about; a reply that uses one the
# conversation language does not use is a drift and must not reach the TTS
_ALL_SCRIPTS = {
    "hi": r"\u0900-\u097F", "ar": r"\u0600-\u06FF", "ru": r"\u0400-\u04FF", "ja": r"\u3040-\u30FF",
    "ko": r"\uAC00-\uD7AF", "zh": r"\u4E00-\u9FFF", "el": r"\u0370-\u03FF", "he": r"\u0590-\u05FF",
    "_bn": r"\u0980-\u09FF", "_pa": r"\u0A00-\u0A7F", "_gu": r"\u0A80-\u0AFF", "_ta": r"\u0B80-\u0BFF",
    "_te": r"\u0C00-\u0C7F", "_kn": r"\u0C80-\u0CFF", "_ml": r"\u0D00-\u0D7F", "_th": r"\u0E00-\u0E7F",
}


def foreign_letters(text: str, lang: str) -> bool:
    """True if `text` contains letters from a script the reply language does not use."""
    allowed = {lang, "zh"} if lang == "ja" else {lang}
    ranges = "".join(r for k, r in _ALL_SCRIPTS.items() if k not in allowed)
    return re.search(f"[{ranges}]", text) is not None


_LANG_NAMES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "ar": "Arabic", "pt": "Portuguese", "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ru": "Russian", "tr": "Turkish", "nl": "Dutch", "pl": "Polish"}
_SPEAKABLE = set(_LANG_NAMES) | {"sv", "da", "fi", "no", "el", "he", "ms", "sw"}


def script_lang(text: str, default: str) -> str:
    for rx, lang in _SCRIPTS:
        if rx.search(text):
            return lang
    return default


def _strip_unspeakable(text: str) -> str:
    """Markdown, emoji and symbols the models would try to read out."""
    text = re.sub(r"[*_`#>\[\]()]+", " ", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class LLM:
    """Streaming chat over any OpenAI-compatible endpoint (Ollama by default)."""

    def __init__(self) -> None:
        self.base = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.key = settings.llm_api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))

    async def warm(self) -> None:
        """Load the model into Ollama's VRAM so the first real turn is not a 2-3 s cold start."""
        try:
            async for _ in self.stream([{"role": "user", "content": "Say OK."}]):
                pass
        except Exception:  # noqa: BLE001
            pass

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
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": 0.6, "max_tokens": 200, "keep_alive": "24h"}
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
        self.speed = 1.0
        self.system = settings.llm_system_prompt
        self.history: list[dict] = []
        self.cancel = threading.Event()
        self.turn_task: asyncio.Task | None = None
        self.meter = None  # set by the server: (chars, audio_s, ttfa_ms, total_ms, lang, kind) -> None
        self._pcm = bytearray()  # live mic frames of the turn being recorded
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
                raw = await self.ws.receive()
                if raw.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(raw.get("code", 1000))
                if raw.get("bytes") is not None:
                    self._pcm += raw["bytes"]
                    continue
                msg = json.loads(raw.get("text") or "{}")
                t = msg.get("type")
                if t == "turn_start":
                    await self._cancel()
                    self._pcm = bytearray()
                elif t == "turn_end":
                    pcm, self._pcm = bytes(self._pcm), bytearray()
                    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    self.cancel = threading.Event()
                    self.turn_task = asyncio.create_task(self._turn({"type": "pcm", "audio": audio}))
                elif t == "config":
                    self.voice = msg.get("voice") or self.voice
                    self.language = msg.get("language") or None
                    if msg.get("speed"):
                        self.speed = max(0.7, min(1.6, float(msg["speed"])))
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
            if msg["type"] in ("turn", "pcm"):
                if self.stt is None:
                    await self.send(type="error", message="speech-to-text is disabled on this server (TTS_STT=0)")
                    return
                if msg["type"] == "pcm":
                    audio = msg["audio"]  # already float32 16 kHz: nothing to decode
                else:
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

            if lang not in _SPEAKABLE:
                lang = "en"
            lang_name = _LANG_NAMES.get(lang, lang)
            log.info("talk heard [%s] %r", lang, user_text)

            # 2. think + 3. speak, overlapped: sentences go to TTS as soon as they complete.
            # The reply language is pinned explicitly; small models drift otherwise.
            # NOTE: do not append instructions to the user turn - a trailing "(answer in
            # English, in its native script)" made Qwen answer in random scripts 6/6.
            system = self.system + f"\nThe user speaks {lang_name}. Always answer in {lang_name}."
            messages = [{"role": "system", "content": system}, *self.history[-12:], {"role": "user", "content": user_text}]
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            speaker = asyncio.create_task(self._speak(sentences, lang, t0, timings, cancel))

            reply = ""
            buf = ""
            for attempt in range(2):
                reply = ""
                buf = ""
                first = True
                drifted = False
                n_sent = 0
                async for tok in self.llm.stream(messages):
                    if cancel.is_set():
                        break
                    if first:
                        first = False
                        timings["llm_first_token_ms"] = timings["llm_first_token_ms"] or round((time.perf_counter() - t0) * 1000)
                    reply += tok
                    # wrong script anywhere in the reply -> restart with a firmer instruction
                    if attempt == 0 and foreign_letters(reply, lang):
                        drifted = True
                        break
                    await self.send(type="token", text=tok)
                    # the first piece of speech goes out at the first clause boundary (or ~12
                    # words), not at the first full stop: that is 200-400 ms off first audio
                    words = len(buf.split())
                    if n_sent == 0 and tok[:1].isspace() and words >= FIRST_CLAUSE_MIN_WORDS and not _TERM.search(buf):
                        if _CLAUSE.search(buf) or words >= FIRST_CLAUSE_MAX_WORDS:
                            piece = buf.strip()
                            await sentences.put(piece if piece[-1] in ",;:—–" else piece + ",")
                            n_sent += 1
                            buf = ""
                    buf += tok
                    if _TERM.search(buf) and len(buf.strip()) > 2:
                        for sent in split_sentences(buf):
                            await sentences.put(sent)
                            n_sent += 1
                        buf = ""
                if drifted:
                    log.warning("talk: LLM answered in the wrong script (%r); retrying with a forced language", reply[:40])
                    messages[-1] = {"role": "user", "content": f"{user_text}\n\nIMPORTANT: reply ONLY in {lang_name}. Do not use any other language or script."}
                    continue
                break
            if foreign_letters(reply, lang):
                # still wrong after the retry: say so instead of babbling through it
                log.warning("talk: LLM drifted twice (%r); speaking a fallback", reply[:40])
                reply = {"hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप दोबारा कह सकते हैं?"}.get(lang, "Sorry, I lost my train of thought. Could you say that again?")
                buf = ""
                await self.send(type="token", text=reply)
                await sentences.put(reply)
            if buf.strip() and not cancel.is_set():
                await sentences.put(buf.strip())
            await sentences.put(None)
            await speaker
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
            log.info("talk reply [%s] %r (first audio %s ms)", lang, reply.strip()[:120], timings["first_audio_ms"])
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
        self._audio_s = 0.0
        self._kind = self.engine.pick(lang)
        params = GenParams()
        while True:
            s = await sentences.get()
            if s is None or cancel.is_set():
                return
            s = _strip_unspeakable(s)
            if not re.search(r"\w", s) or foreign_letters(s, lang):
                continue  # nothing speakable, or a script the models cannot speak
            # each sentence is routed by its own script, so a stray Hindi or
            # Chinese sentence goes to the multilingual model instead of Turbo
            s_lang = script_lang(s, lang)
            kind = self.engine.pick(s_lang)
            conds = await self.registry.ensure(voice, kind)
            gain = settings.gain_turbo if kind == "turbo" else settings.gain_multilingual
            await self.send(type="sentence", text=s)
            for chunk in chunk_text(s):
                q = self.engine.worker.submit_stream(
                    lambda c=chunk, k=kind, cd=conds, l=s_lang, sp=self.speed: self.engine.stream(k, c, cd, l, params, speed=sp), priority=0, cancel=cancel
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
