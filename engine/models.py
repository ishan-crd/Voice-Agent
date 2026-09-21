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
from typing import Any, Callable, Iterator, Literal

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
    priority: tuple[int, float]  # (class, submit time): class 0 = first chunk of a request
    seq: int
    fn: Callable[[], Any] = field(compare=False)
    on_done: Callable[[str, Any], None] = field(compare=False)
    cancel: threading.Event | None = field(compare=False, default=None)


class GpuWorker:
    """Runs callables sequentially on one background thread.

    Jobs are ordered by (priority class, submit time).  The server submits the
    first chunk of every request as class 0 and the rest as class 1, so a new
    caller's first audio jumps ahead of everyone else's long tail; within a
    class it is FIFO, which with the per-request lookahead cap gives
    round-robin-ish fairness across concurrent conversations.
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
        priority: int = 1,
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

        self._q.put(_Job((priority, time.time()), self._next_seq(), fn, on_done, cancel))
        return fut

    def submit_stream(
        self,
        gen_fn: Callable[[], Iterator[Any]],
        *,
        priority: int = 1,
        cancel: threading.Event | None = None,
    ) -> asyncio.Queue:
        """Run a generator on the GPU thread; each yielded item lands in the
        returned queue as soon as it exists.  `None` marks the end, an
        Exception instance marks failure.  Stops early when `cancel` is set."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def push(item: Any) -> None:
            loop.call_soon_threadsafe(q.put_nowait, item)

        def run() -> None:
            try:
                for item in gen_fn():
                    push(item)
                    if cancel is not None and cancel.is_set():
                        break
            except BaseException as e:  # noqa: BLE001
                log.exception("gpu stream failed")
                push(e)
            else:
                push(None)

        def on_done(status: str, payload: Any) -> None:
            if status == "cancelled":
                push(None)

        self._q.put(_Job((priority, time.time()), self._next_seq(), run, on_done, cancel))
        return q

    def run_sync(self, fn: Callable[[], Any]) -> Any:
        """Blocking helper for startup code (no event loop yet)."""
        done = threading.Event()
        box: dict[str, Any] = {}

        def on_done(status: str, payload: Any) -> None:
            box[status] = payload
            done.set()

        self._q.put(_Job((0, time.time()), self._next_seq(), fn, on_done, None))
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
        self.streamers: dict[str, Any] = {}  # kind -> TurboStreamer
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

                model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
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
            if settings.streaming:
                from .streaming import MultilingualStreamer, StreamConfig, TurboStreamer

                cls = TurboStreamer if kind == "turbo" else MultilingualStreamer
                cfg = StreamConfig(
                    first_block=settings.first_block_tokens,
                    t3_dtype=torch.float16 if settings.t3_fp16 else torch.float32,
                    cuda_graph=settings.cuda_graph,
                )
                if kind == "multilingual":
                    # the 10-step CFG vocoder costs ~250 ms per block, so use fewer,
                    # larger blocks: first audio ~450 ms with a comfortable playback margin
                    cfg.first_block = max(settings.first_block_tokens, 20)
                    cfg.block_growth = (20, 30, 50)
                self.streamers[kind] = cls(model, cfg)
                self._trim_ref(self.builtin_conds.get(kind))
                # reference prompts are 5-6 s (250-300 mel frames); capture from there up
                self.streamers[kind].warmup_graphs(prompt_frames=250)
            log.info("loaded %s in %.1fs", kind, time.perf_counter() - t0)

        self._warmup()

    def _trim_ref(self, conds) -> None:
        """Shorten the S3Gen reference prompt: the flow decoder re-encodes it on
        every streamed block, so 10 s of prompt costs ~2x the vocoder time of 5 s."""
        secs = settings.ref_seconds
        if conds is None or secs <= 0:
            return
        g = conds.gen
        n_tok, n_mel = 25 * secs, 50 * secs
        if g["prompt_token"].shape[1] <= n_tok:
            return
        g["prompt_token"] = g["prompt_token"][:, :n_tok]
        g["prompt_token_len"] = torch.tensor([n_tok], device=g["prompt_token"].device)
        g["prompt_feat"] = g["prompt_feat"][:, :n_mel]
        g["prompt_feat_len"] = torch.tensor([n_mel], device=g["prompt_feat"].device)

    def _warmup(self) -> None:
        """The first CUDA call compiles kernels (and captures the T3 graph); do it before real traffic."""
        for kind in self.models:
            conds = self.builtin_conds.get(kind)
            if conds is None:
                continue
            try:
                t0 = time.perf_counter()
                texts = [
                    "Warm up.",
                    "Warm up, this is a longer sentence to settle the kernels and the allocator.",
                    "And one more, with a few clauses, so every early block length has been seen once before real traffic arrives.",
                ]
                for t in texts:
                    self.synthesize(kind, t, conds, "en", GenParams())
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
        if kind in self.streamers:
            self._trim_ref(conds)
        return conds

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def stream(
        self,
        kind: str,
        text: str,
        conds,
        language: str,
        params: GenParams,
        speed: float = 1.0,
    ) -> Iterator[np.ndarray]:
        """Yield float32 mono 24 kHz audio blocks as they are produced.

        Turbo streams at token level (many small blocks per sentence); the
        multilingual model yields one block per call.
        """
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        streamer = self.streamers.get(kind)
        if streamer is not None:
            yield from streamer.stream(text, conds, params, language=language, speed=speed)
            return

        model = self.models[kind]
        model.conds = conds
        wav = model.generate(
            text,
            language_id=language.lower(),
            exaggeration=params.exaggeration,
            cfg_weight=params.cfg_weight,
            temperature=params.temperature,
            repetition_penalty=params.repetition_penalty,
            top_p=params.top_p,
        )
        yield wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def synthesize(self, kind: str, text: str, conds, language: str, params: GenParams) -> np.ndarray:
        """Non-streaming convenience: the whole utterance as one array."""
        blocks = list(self.stream(kind, text, conds, language, params))
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)
