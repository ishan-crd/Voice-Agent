"""Model loading and the single GPU worker thread.

Chatterbox models are not thread-safe and share one GPU, so every call that
touches a model goes through `GpuWorker.submit()`.  Callers get an asyncio
Future back and never block the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch

from .config import settings

log = logging.getLogger("tts.models")

ModelKind = Literal["turbo", "multilingual"]
SAMPLE_RATE = 24_000  # S3GEN_SR for every Chatterbox variant


@dataclass
class GenParams:
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    top_p: float = 0.95
    seed: int | None = None


@dataclass(order=True)
class _Job:
    priority: float
    seq: int
    fn: Callable[[], Any] = field(compare=False)
    on_done: Callable[[str, Any], None] = field(compare=False)
    cancel: threading.Event | None = field(compare=False, default=None)


class GpuWorker:
    """Runs callables sequentially on one background thread.

    Priority is the wall-clock time the *request* started, so the caller that
    has been waiting longest gets its next chunk first.  Combined with the
    per-request lookahead cap in the server this gives round-robin-ish
    fairness across concurrent conversations without starving anyone.
    """

    def __init__(self) -> None:
        self._q: queue.PriorityQueue[_Job] = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self._thread.start()

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        priority: float | None = None,
        cancel: threading.Event | None = None,
    ) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def on_done(status: str, payload: Any) -> None:
            def _apply():
                if fut.done():
                    return
                if status == "ok":
                    fut.set_result(payload)
                elif status == "error":
                    fut.set_exception(payload)
                else:
                    fut.cancel()

            loop.call_soon_threadsafe(_apply)

        self._q.put(_Job(priority or time.time(), self._next_seq(), fn, on_done, cancel))
        return fut

    def run_sync(self, fn: Callable[[], Any]) -> Any:
        """Blocking helper for startup code (no event loop yet)."""
        done = threading.Event()
        box: dict[str, Any] = {}

        def on_done(status: str, payload: Any) -> None:
            box[status] = payload
            done.set()

        self._q.put(_Job(0.0, self._next_seq(), fn, on_done, None))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("ok")

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job.cancel is not None and job.cancel.is_set():
                job.on_done("cancelled", None)
                continue
            try:
                result = job.fn()
            except BaseException as e:  # noqa: BLE001
                log.exception("gpu job failed")
                job.on_done("error", e)
            else:
                job.on_done("ok", result)


class TTSEngine:
    """Holds the warm models and exposes synthesis / conditioning primitives.

    Every public method except `load()` and `pick()` must run on the GPU worker
    thread — wrap calls in `engine.worker.submit(lambda: ...)`.
    """

    def __init__(self) -> None:
        self.worker = GpuWorker()
        self.device = settings.device
        self.models: dict[str, Any] = {}
        self.builtin_conds: dict[str, Any] = {}
        self.sr = SAMPLE_RATE

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        t0 = time.perf_counter()
        self.worker.run_sync(self._load_all)
        log.info("models ready in %.1fs: %s", time.perf_counter() - t0, list(self.models))

    def _load_all(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TTS_DEVICE=cuda but torch.cuda.is_available() is False")

        for kind in settings.model_list:
            t0 = time.perf_counter()
            if kind == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif kind == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model="v3")
            else:
                raise ValueError(f"unknown model kind {kind!r} (use turbo | multilingual)")

            if not settings.watermark:
                try:
                    import perth

                    model.watermarker = perth.DummyWatermarker()
                except Exception:  # noqa: BLE001
                    log.warning("could not disable watermarker; leaving it on")

            self.models[kind] = model
            if getattr(model, "conds", None) is not None:
                self.builtin_conds[kind] = model.conds
            log.info("loaded %s in %.1fs", kind, time.perf_counter() - t0)

        self._warmup()

    def _warmup(self) -> None:
        """The first CUDA call compiles kernels; do it before real traffic."""
        for kind in self.models:
            conds = self.builtin_conds.get(kind)
            if conds is None:
                continue
            try:
                t0 = time.perf_counter()
                self.synthesize(kind, "Warm up.", conds, "en", GenParams())
                log.info("warmed %s in %.2fs", kind, time.perf_counter() - t0)
            except Exception:  # noqa: BLE001
                log.exception("warmup failed for %s", kind)

    # --------------------------------------------------------------- routing
    def pick(self, language: str, preferred: str | None = None) -> str:
        """Choose a model kind for a language: turbo for English when loaded."""
        if preferred in self.models:
            return preferred
        lang = language.lower()
        if lang == "en" and "turbo" in self.models:
            return "turbo"
        if "multilingual" in self.models:
            return "multilingual"
        if "turbo" in self.models:
            if lang != "en":
                raise ValueError(f"language {language!r} needs the multilingual model (TTS_MODELS)")
            return "turbo"
        raise RuntimeError("no models loaded")

    # ---------------------------------------------------------- conditioning
    def prepare_conditionals(self, kind: str, wav_path: str, exaggeration: float = 0.5):
        """Encode a reference clip into speaker conditionals for `kind`."""
        model = self.models[kind]
        model.prepare_conditionals(wav_path, exaggeration=exaggeration)
        conds = model.conds
        model.conds = self.builtin_conds.get(kind, conds)
        return conds

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def synthesize(
        self,
        kind: str,
        text: str,
        conds,
        language: str,
        params: GenParams,
    ) -> np.ndarray:
        """Returns float32 mono audio at 24 kHz."""
        model = self.models[kind]
        model.conds = conds
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        if kind == "turbo":
            wav = model.generate(
                text,
                temperature=params.temperature,
                repetition_penalty=params.repetition_penalty,
                top_p=params.top_p,
            )
        else:
            wav = model.generate(
                text,
                language_id=language.lower(),
                exaggeration=params.exaggeration,
                cfg_weight=params.cfg_weight,
                temperature=params.temperature,
                repetition_penalty=params.repetition_penalty,
                top_p=params.top_p,
            )
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
