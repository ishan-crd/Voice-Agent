"""Measure time-to-first-byte and total time against the running server.

    .venv\\Scripts\\python scripts\\bench_ttfb.py [--url http://127.0.0.1:8000] [--voice default]
        [--format pcm] [--n 5] [--concurrency 1] [--text "..."] [--save out/bench.wav]
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

DEFAULT_TEXT = (
    "Hello, am I speaking to Amit? Great. We found multiple job profiles based on your CV, "
    "and I wanted to walk you through them. Do you have a couple of minutes right now?"
)


async def one(client: httpx.AsyncClient, url: str, body: dict, save: Path | None) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    ttfb = None
    n = 0
    buf = bytearray()
    async with client.stream("POST", f"{url}/v1/audio/speech", json=body) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if ttfb is None and chunk:
                ttfb = (time.perf_counter() - t0) * 1000
            n += len(chunk)
            if save:
                buf.extend(chunk)
    total = (time.perf_counter() - t0) * 1000
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        if body["response_format"] == "pcm":
            import numpy as np
            import soundfile as sf

            sr = body.get("sample_rate") or 24000
            sf.write(save, np.frombuffer(bytes(buf), dtype="<i2"), sr)
        else:
            save.write_bytes(bytes(buf))
    return ttfb or total, total, n


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--voice", default="default")
    ap.add_argument("--format", default="pcm")
    ap.add_argument("--language", default=None)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    body = {"model": "chatterbox", "input": args.text, "voice": args.voice, "response_format": args.format}
    if args.language:
        body["language"] = args.language
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        h = (await client.get(f"{args.url}/health")).json()
        print(f"server: models={h['models']} voices={h['voices']} gpu={h.get('gpu', {}).get('name')}")
        await one(client, args.url, body, None)  # warm the route

        results = []
        for round_ in range(args.n):
            save = Path(args.save) if (args.save and round_ == 0) else None
            batch = await asyncio.gather(*[one(client, args.url, body, save if i == 0 else None) for i in range(args.concurrency)])
            results.extend(batch)
            for ttfb, total, n in batch:
                secs = n / 2 / (body.get("sample_rate") or 24000) if args.format == "pcm" else float("nan")
                print(f"  ttfb={ttfb:6.0f} ms  total={total:6.0f} ms  bytes={n:8d}  audio={secs:4.1f}s")

    ttfbs = [r[0] for r in results]
    totals = [r[1] for r in results]
    print(
        f"\n{args.n}x{args.concurrency} {args.format} voice={args.voice}: "
        f"ttfb p50={statistics.median(ttfbs):.0f} ms  p95={sorted(ttfbs)[int(len(ttfbs) * 0.95) - 1 if len(ttfbs) > 1 else 0]:.0f} ms  "
        f"max={max(ttfbs):.0f} ms | total p50={statistics.median(totals):.0f} ms"
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
