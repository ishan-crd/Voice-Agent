"""Voice registry: reference clips -> cached speaker conditionals per model.

Layout (see voices/README.md):

    voices/<voice_id>.wav            reference clip
    voices/voices.json               optional metadata per voice_id
    voices/.cache/<voice_id>.<model>.<fingerprint>.pt   cached Conditionals

Conditionals are computed once per (clip, model) on the GPU worker and kept in
RAM; the on-disk cache makes restarts instant.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .models import TTSEngine

log = logging.getLogger("tts.voices")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# OpenAI voice names map to the default voice so unchanged clients keep working.
OPENAI_ALIASES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"}


@dataclass
class Voice:
    id: str
    path: Path | None  # None for the model's built-in voice
    description: str = ""
    language: str = "en"
    exaggeration: float = 0.5
    builtin: bool = False
    conds: dict[str, Any] = field(default_factory=dict)  # model kind -> Conditionals
    created_at: float = 0.0

    def to_public(self) -> dict[str, Any]:
        return {
            "voice_id": self.id,
            "name": self.id,
            "description": self.description,
            "language": self.language,
            "exaggeration": self.exaggeration,
            "builtin": self.builtin,
            "ready_for": sorted(self.conds),
            "created_at": int(self.created_at),
        }


class VoiceRegistry:
    def __init__(self, engine: TTSEngine, root: Path | None = None) -> None:
        self.engine = engine
        self.root = Path(root or settings.voices_dir)
        self.cache_dir = self.root / ".cache"
        self.voices: dict[str, Voice] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ---------------------------------------------------------------- setup
    def load(self) -> None:
        """Scan the voices dir and precompute conditionals for every model (blocking; startup only)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        meta = self._read_meta()

        for kind, conds in self.engine.builtin_conds.items():
            v = self.voices.setdefault(
                "default",
                Voice(id="default", path=None, description="Chatterbox built-in voice", builtin=True),
            )
            v.conds[kind] = conds

        for p in sorted(self.root.iterdir()):
            if p.suffix.lower() not in AUDIO_EXTS or p.name.startswith("."):
                continue
            vid = p.stem
            m = meta.get(vid, {})
            self.voices[vid] = Voice(
                id=vid,
                path=p,
                description=m.get("description", ""),
                language=m.get("language", "en"),
                exaggeration=float(m.get("exaggeration", 0.5)),
                created_at=p.stat().st_mtime,
            )

        for v in self.voices.values():
            if v.builtin:
                continue
            for kind in self.engine.models:
                try:
                    v.conds[kind] = self.engine.worker.run_sync(lambda v=v, k=kind: self._compute(v, k))
                except Exception:  # noqa: BLE001
                    log.exception("failed to prepare voice %s for %s", v.id, kind)
        log.info("voices: %s", ", ".join(sorted(self.voices)) or "(none)")

    def _read_meta(self) -> dict[str, dict]:
        f = self.root / "voices.json"
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.exception("bad voices.json; ignoring")
            return {}

    def _write_meta(self) -> None:
        data = {
            v.id: {"description": v.description, "language": v.language, "exaggeration": v.exaggeration}
            for v in self.voices.values()
            if not v.builtin
        }
        (self.root / "voices.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ----------------------------------------------------------- conditionals
    def _fingerprint(self, v: Voice) -> str:
        st = v.path.stat()
        raw = f"{v.path.name}:{st.st_size}:{int(st.st_mtime)}:{v.exaggeration}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _compute(self, v: Voice, kind: str):
        """GPU-thread only: load from disk cache or run prepare_conditionals."""
        cache = self.cache_dir / f"{v.id}.{kind}.{self._fingerprint(v)}.pt"
        if cache.exists():
            try:
                from chatterbox.tts_turbo import Conditionals as TurboConds
                from chatterbox.mtl_tts import Conditionals as MtlConds

                cls = TurboConds if kind == "turbo" else MtlConds
                conds = cls.load(cache, map_location=self.engine.device)
                log.info("voice %s/%s: cache hit", v.id, kind)
                return conds
            except Exception:  # noqa: BLE001
                log.warning("voice %s/%s: cache unreadable, recomputing", v.id, kind)

        t0 = time.perf_counter()
        conds = self.engine.prepare_conditionals(kind, str(v.path), exaggeration=v.exaggeration)
        try:
            conds.save(cache)
        except Exception:  # noqa: BLE001
            log.warning("could not write cache %s", cache)
        log.info("voice %s/%s: prepared in %.2fs", v.id, kind, time.perf_counter() - t0)
        return conds

    async def ensure(self, voice: Voice, kind: str):
        """Return conditionals for (voice, model), computing on the GPU worker if needed."""
        if kind in voice.conds:
            return voice.conds[kind]
        lock = self._locks.setdefault((voice.id, kind), asyncio.Lock())
        async with lock:
            if kind not in voice.conds:
                voice.conds[kind] = await self.engine.worker.submit(lambda: self._compute(voice, kind))
        return voice.conds[kind]

    # ------------------------------------------------------------------ crud
    def resolve(self, voice_id: str | None) -> Voice:
        vid = (voice_id or settings.default_voice).strip()
        if vid in self.voices:
            return self.voices[vid]
        if vid.lower() in OPENAI_ALIASES or vid == "":
            fallback = settings.default_voice if settings.default_voice in self.voices else "default"
            if fallback in self.voices:
                return self.voices[fallback]
        raise KeyError(vid)

    def list(self) -> list[dict[str, Any]]:
        return [v.to_public() for v in self.voices.values()]

    async def add(
        self,
        voice_id: str,
        src: Path,
        *,
        description: str = "",
        language: str = "en",
        exaggeration: float = 0.5,
        overwrite: bool = False,
    ) -> Voice:
        vid = _safe_id(voice_id)
        if vid in self.voices and not (overwrite and not self.voices[vid].builtin):
            raise FileExistsError(vid)
        dst = self.root / f"{vid}{src.suffix.lower()}"
        shutil.move(str(src), dst)
        v = Voice(
            id=vid,
            path=dst,
            description=description,
            language=language,
            exaggeration=exaggeration,
            created_at=time.time(),
        )
        self.voices[vid] = v
        self._write_meta()
        for kind in self.engine.models:
            await self.ensure(v, kind)
        return v

    def delete(self, voice_id: str) -> None:
        v = self.voices.get(voice_id)
        if v is None:
            raise KeyError(voice_id)
        if v.builtin:
            raise PermissionError("cannot delete the built-in voice")
        del self.voices[voice_id]
        if v.path and v.path.exists():
            v.path.unlink()
        for f in self.cache_dir.glob(f"{voice_id}.*.pt"):
            f.unlink(missing_ok=True)
        self._write_meta()


def _safe_id(raw: str) -> str:
    vid = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw.strip())[:64]
    if not vid or vid.startswith("."):
        raise ValueError("invalid voice_id")
    return vid
