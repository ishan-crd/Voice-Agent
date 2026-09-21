"""Where does the flow decoder's per-block time go, and does fp16 autocast help?"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_MODELS", "turbo")

import torch  # noqa: E402

from engine.models import TTSEngine  # noqa: E402

engine = TTSEngine()
engine.load()
model = engine.models["turbo"]
flow = model.s3gen.flow
ref = engine.builtin_conds["turbo"].gen
dev = model.device


def run(n_tok: int, autocast: bool) -> float:
    tokens = torch.randint(0, 6000, (1, n_tok), device=dev)
    noise = torch.randn(1, 80, 2 * n_tok, device=dev)
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else torch.no_grad()
    with torch.inference_mode(), ctx:
        for _ in range(2):  # warm
            flow.inference(token=tokens, token_len=torch.tensor([n_tok], device=dev), finalize=True, n_timesteps=2, noised_mels=noise, meanflow=model.s3gen.meanflow, **ref)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            flow.inference(token=tokens, token_len=torch.tensor([n_tok], device=dev), finalize=True, n_timesteps=2, noised_mels=noise, meanflow=model.s3gen.meanflow, **ref)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 5 * 1000


# component split (fp32, 50 tokens)
T = {}
enc, est = flow.encoder.forward, flow.decoder.forward


def timed(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize(); T[name] = T.get(name, 0) + (time.perf_counter() - t0) * 1000; return r
    return w


flow.encoder.forward = timed("encoder", enc)
flow.decoder.forward = timed("cfm_decoder", est)
run(50, False); T.clear(); run(50, False)
print("fp32 split (x5 runs):", {k: f"{v / 5:.0f}ms" for k, v in T.items()})
flow.encoder.forward, flow.decoder.forward = enc, est

for n in (10, 50, 150):
    print(f"tokens={n:3d}  fp32={run(n, False):.0f}ms  autocast_fp16={run(n, True):.0f}ms")
