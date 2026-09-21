"""Can the CFM estimator run on zero-padded, masked buckets (for CUDA-graph capture) without changing valid frames?"""
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
cfm = flow.decoder
est = cfm.estimator
dev = model.device
ref = engine.builtin_conds["turbo"].gen

# realistic inputs: run the encoder path once to get mu/spks/cond for 50 tokens
n_tok = 50
tokens = torch.randint(0, 6000, (1, n_tok), device=dev)
with torch.inference_mode():
    # replicate flow.inference up to the decoder call
    import torch.nn.functional as F
    from chatterbox.models.s3gen.utils.mask import make_pad_mask

    emb = F.normalize(torch.atleast_2d(ref["embedding"]), dim=1)
    emb = flow.spk_embed_affine_layer(emb)
    tok = torch.cat([ref["prompt_token"], tokens], dim=1)
    tok_len = ref["prompt_token_len"] + n_tok
    mask = (~make_pad_mask(tok_len)).unsqueeze(-1).to(emb)
    h, h_masks = flow.encoder(flow.input_embedding(tok.long()) * mask, tok_len)
    h = flow.encoder_proj(h)
    mel_len1 = ref["prompt_feat"].shape[1]
    T = h.shape[1]
    conds = torch.zeros([1, T, 80], device=dev)
    conds[:, :mel_len1] = ref["prompt_feat"]
    conds = conds.transpose(1, 2)
    mu = h.transpose(1, 2).contiguous()
    mk = torch.ones(1, 1, T, device=dev)
    z = torch.randn(1, 80, T, device=dev)
    t_span = torch.linspace(0, 1, 3, device=dev)

    def euler(x, mu, mk, conds, L):
        for t, r in zip(t_span[:-1], t_span[1:]):
            t, r = t[None], r[None]
            dxdt = est(x, mask=mk, mu=mu, t=t, spks=emb, cond=conds, r=r)
            x = x + (r - t) * dxdt
        return x

    ref_out = euler(z.clone(), mu, mk, conds, T)

    # padded to bucket
    B = ((T + 63) // 64) * 64
    pad = B - T
    P = lambda a: F.pad(a, (0, pad))  # noqa: E731
    pad_out = euler(P(z), P(mu), P(mk), P(conds), B)
    diff = (pad_out[..., :T] - ref_out).abs()
    print(f"T={T} bucket={B}: max|diff| valid frames = {diff.max().item():.2e}, mean = {diff.mean().item():.2e}, ref scale = {ref_out.abs().mean().item():.2f}")
    # tail-only diff (last 20 frames are the ones nearest the padding)
    print(f"   last 20 frames max|diff| = {diff[..., -20:].max().item():.2e}")

    # timing: eager vs CUDA graph of the 2-step euler on the bucket
    xs, mus, mks, cs = P(z).clone(), P(mu).clone(), P(mk).clone(), P(conds).clone()
    out_buf = torch.empty_like(xs)

    def body():
        out_buf.copy_(euler(xs, mus, mks, cs, B))

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): body()
    torch.cuda.synchronize(); print(f"eager 2-step euler @ {B} frames: {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); print(f"graph 2-step euler @ {B} frames: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms")
    print("graph == eager:", (out_buf[..., :T] - pad_out[..., :T]).abs().max().item())
