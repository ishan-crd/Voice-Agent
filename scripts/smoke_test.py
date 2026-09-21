"""Load the engine in-process (no HTTP), synthesize a sentence, save a wav.

    .venv\\Scripts\\python scripts\\smoke_test.py [--models turbo] [--voice path.wav] [--lang hi]

Prints per-chunk generation time so you can see raw model latency before
any HTTP/encoding overhead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="override TTS_MODELS, e.g. turbo")
    ap.add_argument("--voice", default=None, help="reference clip to clone (default: built-in voice)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--text", default="Hello, am I speaking to Amit? Great. We found multiple job profiles based on your CV, and I wanted to walk you through them.")
    ap.add_argument("--out", default="out/smoke.wav")
    args = ap.parse_args()
    if args.models:
        os.environ["TTS_MODELS"] = args.models

    import numpy as np
    import soundfile as sf

    from engine.chunking import chunk_text
    from engine.models import GenParams, TTSEngine

    engine = TTSEngine()
    engine.load()

    kind = engine.pick(args.lang)
    if args.voice:
        t0 = time.perf_counter()
        conds = engine.worker.run_sync(lambda: engine.prepare_conditionals(kind, args.voice))
        print(f"prepare_conditionals({args.voice}) -> {time.perf_counter() - t0:.2f}s")
    else:
        conds = engine.builtin_conds[kind]

    chunks = chunk_text(args.text)
    print(f"model={kind} lang={args.lang} chunks={len(chunks)}")
    pieces = []
    t_req = time.perf_counter()
    for i, c in enumerate(chunks):
        t0 = time.perf_counter()
        audio = engine.worker.run_sync(lambda c=c: engine.synthesize(kind, c, conds, args.lang, GenParams()))
        dt = time.perf_counter() - t0
        secs = len(audio) / engine.sr
        flag = "  <-- first audio" if i == 0 else ""
        print(f"  [{i}] {dt * 1000:6.0f} ms gen | {secs:4.1f}s audio | RTF {dt / secs:.2f} | {c!r}{flag}")
        pieces.append(audio)
    total = time.perf_counter() - t_req
    audio = np.concatenate(pieces)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, engine.sr)
    print(f"total {total * 1000:.0f} ms for {len(audio) / engine.sr:.1f}s of audio -> {args.out}")


if __name__ == "__main__":
    main()
