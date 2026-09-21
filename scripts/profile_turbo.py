"""Break Turbo's generate() into T3 (tokens), S3Gen (vocoder) and watermark time."""
from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_MODELS", "turbo")

import torch  # noqa: E402

from engine.models import GenParams, TTSEngine  # noqa: E402

engine = TTSEngine()
engine.load()
model = engine.models["turbo"]
timings: dict[str, float] = {}


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0) + (time.perf_counter() - t0)
        return r

    return w


model.t3.inference_turbo = timed("t3_tokens", model.t3.inference_turbo)
model.s3gen.inference = timed("s3gen_vocoder", model.s3gen.inference)
model.watermarker.apply_watermark = timed("watermark", model.watermarker.apply_watermark)
model.norm_loudness = timed("loudness", model.norm_loudness)

texts = ["Hello, am I speaking to Amit?", "Hello, am I speaking to Amit? Great.", "We found multiple job profiles based on your CV, and I wanted to walk you through them."]
for text in texts:
    for run in range(2):
        timings.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wav = engine.worker.run_sync(lambda: engine.synthesize("turbo", text, engine.builtin_conds["turbo"], "en", GenParams()))
        total = time.perf_counter() - t0
        secs = len(wav) / engine.sr
        parts = "  ".join(f"{k}={v * 1000:.0f}ms" for k, v in timings.items())
        print(f"run{run} total={total * 1000:.0f}ms audio={secs:.1f}s | {parts} | {text[:40]!r}")
print("dtype:", next(model.t3.parameters()).dtype, "| s3gen:", next(model.s3gen.parameters()).dtype)
