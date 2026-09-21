"""Per-stage timing inside the streamer: T3 per-token, flow per block, HiFT per block."""
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
from engine.streaming import TurboStreamer  # noqa: E402

engine = TTSEngine()
engine.load()
model = engine.models["turbo"]
conds = engine.builtin_conds["turbo"]
print("ref prompt tokens:", conds.gen["prompt_token"].shape, "prompt mel:", conds.gen["prompt_feat"].shape)
print("t3 dtype:", next(model.t3.parameters()).dtype, "tfmr:", type(model.t3.tfmr).__name__)

T: dict[str, list[float]] = {"flow": [], "hift": [], "tfmr": []}
import time as _t


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[name].append((time.perf_counter() - t0) * 1000)
        return r

    return w


model.s3gen.flow.inference = timed("flow", model.s3gen.flow.inference)
model.s3gen.mel2wav.inference = timed("hift", model.s3gen.mel2wav.inference)
model.t3.speech_head.forward = timed("tfmr", model.t3.speech_head.forward)  # cheap proxy for one decode step (graph-safe: captured already)

streamer = engine.streamers["turbo"]
text = "We found multiple job profiles based on your CV, and I wanted to walk you through them."
for run in range(2):
    for k in T:
        T[k].clear()
    t0 = time.perf_counter()
    n = 0
    for b in streamer.stream(text, conds, GenParams()):
        n += len(b)
    total = (time.perf_counter() - t0) * 1000
    tf = T["tfmr"]
    print(
        f"run{run}: total={total:.0f}ms audio={n / 24000:.1f}s | tfmr: prefill={tf[0]:.0f}ms, "
        f"{len(tf) - 1} steps avg {sum(tf[1:]) / max(1, len(tf) - 1):.1f}ms | "
        f"flow: {[f'{x:.0f}' for x in T['flow']]} | hift: {[f'{x:.0f}' for x in T['hift']]}"
    )
