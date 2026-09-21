"""Measure time-to-first-audio with the token-level streamer vs stock generate()."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_MODELS", "turbo")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from engine.models import GenParams, TTSEngine  # noqa: E402

engine = TTSEngine()
engine.load()
model = engine.models["turbo"]
conds = engine.builtin_conds["turbo"]
streamer = engine.streamers["turbo"]

texts = [
    "Hello, am I speaking to Amit?",
    "We found multiple job profiles based on your CV, and I wanted to walk you through them.",
]
Path("out").mkdir(exist_ok=True)
for text in texts:
    for run in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        blocks, marks = [], []
        for b in streamer.stream(text, conds, GenParams()):
            marks.append((time.perf_counter() - t0) * 1000)
            blocks.append(b)
        total = (time.perf_counter() - t0) * 1000
        audio = np.concatenate(blocks)
        secs = len(audio) / engine.sr
        # playback margin: if the client starts playing at TTFA, does every later
        # block arrive before the audio already delivered runs out?
        margin = float("inf")
        buffered = 0.0
        for i, (b, t) in enumerate(zip(blocks, marks)):
            if i > 0:
                margin = min(margin, buffered - (t - marks[0]))
            buffered += len(b) / engine.sr * 1000
        print(
            f"stream run{run}: TTFA={marks[0]:.0f}ms  blocks={len(blocks)}  total={total:.0f}ms  "
            f"audio={secs:.1f}s  RTF={total / 1000 / secs:.2f}  min_margin={margin:+.0f}ms  "
            f"block_t=[{', '.join(f'{m:.0f}' for m in marks)}]  {text[:30]!r}"
        )
    sf.write(f"out/stream_{len(text)}.wav", audio, engine.sr)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wav = engine.synthesize("turbo", text, conds, "en", GenParams())
    total = (time.perf_counter() - t0) * 1000
    print(f"stock generate: {total:.0f}ms for {len(wav) / engine.sr:.1f}s audio")
    sf.write(f"out/stock_{len(text)}.wav", wav, engine.sr)
