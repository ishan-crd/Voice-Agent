"""Prototype: T3 decode step with StaticCache + CUDA graph capture. Measures ms/token."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_MODELS", "turbo")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import StaticCache  # noqa: E402

from engine.models import TTSEngine  # noqa: E402

engine = TTSEngine()
engine.load()
model = engine.models["turbo"]
t3 = model.t3
dev = model.device
conds = engine.builtin_conds["turbo"]

text = "We found multiple job profiles based on your CV, and I wanted to walk you through them."
from chatterbox.tts_turbo import punc_norm  # noqa: E402

text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(dev)
start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
embeds, _ = t3.prepare_input_embeds(t3_cond=conds.t3, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
L = embeds.shape[1]
print(f"prefill len={L}, hidden={embeds.shape[-1]}, layers={t3.cfg.num_hidden_layers if hasattr(t3.cfg,'num_hidden_layers') else t3.cfg.n_layer}")

MAX = 1536


def bench(label, step_fn, n=100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tok = torch.tensor([[6000]], device=dev)
    for i in range(n):
        logits = step_fn(tok, L + i)
        tok = torch.multinomial(F.softmax(logits / 0.8, dim=-1), 1)
        _ = tok.item()  # host sync like the real loop
    torch.cuda.synchronize()
    print(f"{label}: {(time.perf_counter() - t0) / n * 1000:.2f} ms/token")


# -------- 1. baseline: DynamicCache eager (what inference_turbo does)
with torch.inference_mode():
    out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past = out.past_key_values

    def step_dyn(tok, pos):
        global past
        o = t3.tfmr(inputs_embeds=t3.speech_emb(tok), past_key_values=past, use_cache=True)
        past = o.past_key_values
        return t3.speech_head(o[0])[:, -1, :]

    bench("dynamic cache fp32 eager", step_dyn)

# -------- 2. StaticCache eager
with torch.inference_mode():
    cache = StaticCache(config=t3.cfg, max_cache_len=MAX)
    t3.tfmr(inputs_embeds=embeds, past_key_values=cache, use_cache=True, cache_position=torch.arange(L, device=dev))

    def step_static(tok, pos):
        o = t3.tfmr(
            inputs_embeds=t3.speech_emb(tok),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([pos], device=dev),
        )
        return t3.speech_head(o[0])[:, -1, :]

    bench("static cache fp32 eager", step_static)

# -------- 3. StaticCache + CUDA graph
with torch.inference_mode():
    cache = StaticCache(config=t3.cfg, max_cache_len=MAX)
    t3.tfmr(inputs_embeds=embeds, past_key_values=cache, use_cache=True, cache_position=torch.arange(L, device=dev))

    s_tok = torch.tensor([[6000]], device=dev)
    s_pos = torch.tensor([L], device=dev)
    s_logits = torch.empty((1, t3.hp.speech_tokens_dict_size), device=dev)

    def graph_body():
        o = t3.tfmr(inputs_embeds=t3.speech_emb(s_tok), past_key_values=cache, use_cache=True, cache_position=s_pos)
        s_logits.copy_(t3.speech_head(o[0])[:, -1, :])

    # warm up on a side stream (required before capture), then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            graph_body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_body()
    print("captured")

    def step_graph(tok, pos):
        s_tok.copy_(tok)
        s_pos.fill_(pos)
        g.replay()
        return s_logits

    bench("static cache + CUDA graph fp32", step_graph)

    # sanity: graph output == eager output for same input/pos
    cache2 = StaticCache(config=t3.cfg, max_cache_len=MAX)
    t3.tfmr(inputs_embeds=embeds, past_key_values=cache2, use_cache=True, cache_position=torch.arange(L, device=dev))
    cache.reset()
    t3.tfmr(inputs_embeds=embeds, past_key_values=cache, use_cache=True, cache_position=torch.arange(L, device=dev))
    tok = torch.tensor([[1234]], device=dev)
    ref = t3.speech_head(t3.tfmr(inputs_embeds=t3.speech_emb(tok), past_key_values=cache2, use_cache=True, cache_position=torch.tensor([L], device=dev))[0])[:, -1, :]
    got = step_graph(tok, L).clone()
    print("max |graph - eager| =", (got - ref).abs().max().item())

# -------- 4. where is the rest? replay-only, then fp16
with torch.inference_mode():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): g.replay()
    torch.cuda.synchronize(); print(f"graph replay only fp32: {(time.perf_counter()-t0)/100*1000:.2f} ms")

    t3.half()
    cache16 = StaticCache(config=t3.cfg, max_cache_len=MAX)
    emb16 = embeds.half()
    t3.tfmr(inputs_embeds=emb16, past_key_values=cache16, use_cache=True, cache_position=torch.arange(L, device=dev))
    s_logits16 = torch.empty((1, t3.hp.speech_tokens_dict_size), device=dev, dtype=torch.float16)
    def body16():
        o = t3.tfmr(inputs_embeds=t3.speech_emb(s_tok), past_key_values=cache16, use_cache=True, cache_position=s_pos)
        s_logits16.copy_(t3.speech_head(o[0])[:, -1, :])
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): body16()
    torch.cuda.current_stream().wait_stream(s)
    g16 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g16): body16()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): g16.replay()
    torch.cuda.synchronize(); print(f"graph replay only fp16: {(time.perf_counter()-t0)/100*1000:.2f} ms")
    def step_graph16(tok, pos):
        s_tok.copy_(tok); s_pos.fill_(pos); g16.replay(); return s_logits16.float()
    bench("static cache + CUDA graph fp16 (+sampling)", step_graph16)
    print("fp16 logits finite:", torch.isfinite(s_logits16).all().item())
