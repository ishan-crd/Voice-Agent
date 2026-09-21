"""Token-level streaming for Chatterbox Turbo.

The stock `generate()` runs the T3 decoder to completion, then vocodes the
whole token sequence.  Here the T3 loop yields speech tokens in growing
blocks and each block is vocoded immediately (CosyVoice2-style chunked
flow-matching + HiFT with source cache and a short crossfade), so the first
audio leaves the GPU after ~10 tokens instead of after the whole sentence.

Only Turbo is supported: its T3 decode loop is plain PyTorch, while the
multilingual model goes through HF `generate()` with CFG.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .models import GenParams

log = logging.getLogger("tts.streaming")

S3GEN_SIL = 4299
SPEECH_VOCAB = 6561
TOKEN_MEL_RATIO = 2  # 25 tok/s -> 50 mel/s
MEL_HOP = 480  # 24 kHz / 50
MEL_CACHE_FRAMES = 8
SOURCE_CACHE = MEL_CACHE_FRAMES * MEL_HOP
MAX_MEL_FRAMES = 2 * 1200  # 48 s, more than T3's max_gen_len


@dataclass
class StreamConfig:
    first_block: int = 12  # tokens before the first vocoder pass (~0.2 s emitted after lookahead+cache)
    block_growth: tuple[int, ...] = (6, 10, 16, 25, 40)  # subsequent block sizes, then `max_block`
    max_block: int = 50
    max_gen_len: int = 1000
    max_cache_len: int = 1536  # conditioning + text + generated speech tokens
    t3_dtype: torch.dtype = torch.float16
    cuda_graph: bool = True
    cfm_bucket: int = 64  # mel frames; CFM decoder graphs are captured per bucket
    cfm_max_frames: int = 1216  # longer sequences fall back to eager


def _chunk_mask_capture_safe(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size, num_decoding_left_chunks, *a, **k):
    """Drop-in for `add_optional_chunk_mask` in the non-chunked mode the CFM
    decoder uses: the original ends with a `.item()` host sync that forbids
    CUDA-graph capture and is a no-op for a (B,1,L) key mask."""
    if use_dynamic_chunk or static_chunk_size > 0:
        from chatterbox.models.s3gen.utils.mask import add_optional_chunk_mask

        return add_optional_chunk_mask(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size, num_decoding_left_chunks, *a, **k)
    return masks


class CfmGraphs:
    """Two-step mean-flow Euler solve captured as one CUDA graph per length bucket.

    The estimator (U-Net + transformer blocks) is launch-bound: ~135 ms per
    call at any length in eager mode, ~15-25 ms replayed.  Inputs are padded
    with a zero mask to the bucket length; the padded frames are masked out of
    attention and multiplied away in the conv blocks, so valid frames match
    the unpadded result to ~1e-3.
    """

    def __init__(self, cfm, device, bucket: int, max_frames: int) -> None:
        self.cfm = cfm
        self.est = cfm.estimator
        self.device = device
        self.bucket = bucket
        self.max_frames = max_frames
        self.t_span = torch.linspace(0, 1, 3, device=device)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, dict[str, torch.Tensor]]] = {}
        self._pool = None

    def euler(self, x, mu, mask, spks, cond) -> torch.Tensor:
        for t, r in zip(self.t_span[:-1], self.t_span[1:]):
            t, r = t[None], r[None]
            dxdt = self.est(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            x = x + (r - t) * dxdt
        return x

    def _bucket_for(self, T: int) -> int | None:
        B = -(-T // self.bucket) * self.bucket
        return B if B <= self.max_frames else None

    @torch.inference_mode()
    def capture(self, B: int, spks_dim: int) -> None:
        if B in self.graphs:
            return
        bufs = {
            "x": torch.zeros(1, 80, B, device=self.device),
            "mu": torch.zeros(1, 80, B, device=self.device),
            "mask": torch.zeros(1, 1, B, device=self.device),
            "cond": torch.zeros(1, 80, B, device=self.device),
            "spks": torch.zeros(1, spks_dim, device=self.device),
            "out": torch.zeros(1, 80, B, device=self.device),
        }
        bufs["mask"][..., : B // 2].fill_(1.0)  # some valid frames so the warm-up is representative

        def body():
            bufs["out"].copy_(self.euler(bufs["x"], bufs["mu"], bufs["mask"], bufs["spks"], bufs["cond"]))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            body()
        self.graphs[B] = (g, bufs)

    @torch.inference_mode()
    def capture_all(self, min_frames: int, spks_dim: int) -> None:
        import time

        t0 = time.perf_counter()
        B = self._bucket_for(min_frames) or self.bucket
        while B <= self.max_frames:
            self.capture(B, spks_dim)
            B += self.bucket
        log.info("captured %d CFM decoder graphs (%d..%d frames) in %.1fs", len(self.graphs), min(self.graphs), max(self.graphs), time.perf_counter() - t0)

    @torch.inference_mode()
    def solve(self, x, mu, mask, spks, cond) -> torch.Tensor:
        T = x.shape[-1]
        B = self._bucket_for(T)
        if B is None or B not in self.graphs:
            return self.euler(x, mu, mask, spks, cond)
        g, b = self.graphs[B]
        pad = B - T
        for k, v in (("x", x), ("mu", mu), ("mask", mask), ("cond", cond)):
            b[k][..., :T].copy_(v)
            if pad:
                b[k][..., T:].zero_()
        b["spks"].copy_(spks)
        g.replay()
        return b["out"][..., :T].clone()


class TurboStreamer:
    def __init__(self, model, cfg: StreamConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or StreamConfig()
        self.device = model.device
        self.hift = model.s3gen.mel2wav
        self.flow = model.s3gen.flow
        self.window = torch.hamming_window(2 * SOURCE_CACHE, device=self.device)
        self.trim_fade = model.s3gen.trim_fade

        t3 = model.t3
        if self.cfg.t3_dtype != next(t3.parameters()).dtype:
            t3.to(self.cfg.t3_dtype)
        self.t3_dtype = self.cfg.t3_dtype
        self._cache = None
        self._graph = None
        self._s_tok = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._s_pos = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._s_logits = torch.empty((1, t3.hp.speech_tokens_dict_size), device=self.device, dtype=self.t3_dtype)

        self.cfm: CfmGraphs | None = None
        if self.cfg.cuda_graph and self.device != "cpu":
            import chatterbox.models.s3gen.decoder as _dec

            _dec.add_optional_chunk_mask = _chunk_mask_capture_safe
            self.cfm = CfmGraphs(self.flow.decoder, self.device, self.cfg.cfm_bucket, self.cfg.cfm_max_frames)

    def warmup_graphs(self, prompt_frames: int) -> None:
        """Capture every CFM bucket that a stream can hit; call once at startup."""
        if self.cfm is not None and not self.cfm.graphs:
            spks_dim = self.flow.spk_embed_affine_layer.out_features
            self.cfm.capture_all(prompt_frames + TOKEN_MEL_RATIO * self.cfg.first_block, spks_dim)

    # ------------------------------------------------------- T3 decode step
    def _decode_step_eager(self) -> None:
        t3 = self.model.t3
        out = t3.tfmr(
            inputs_embeds=t3.speech_emb(self._s_tok),
            past_key_values=self._cache,
            use_cache=True,
            cache_position=self._s_pos,
        )
        self._s_logits.copy_(t3.speech_head(out[0])[:, -1, :])

    @torch.inference_mode()
    def _prefill(self, embeds: torch.Tensor) -> None:
        """Reset the static KV cache and run the conditioning + text prefix through it."""
        from transformers import StaticCache

        t3 = self.model.t3
        if self._cache is None:
            self._cache = StaticCache(config=t3.cfg, max_cache_len=self.cfg.max_cache_len)
        else:
            self._cache.reset()
        L = embeds.shape[1]
        out = t3.tfmr(
            inputs_embeds=embeds.to(self.t3_dtype),
            past_key_values=self._cache,
            use_cache=True,
            cache_position=torch.arange(L, device=self.device),
        )
        self._s_logits.copy_(t3.speech_head(out[0][:, -1:])[:, -1, :])

    @torch.inference_mode()
    def _capture(self) -> None:
        """Capture one decode step as a CUDA graph (shape-static thanks to StaticCache)."""
        if not self.cfg.cuda_graph or self._graph is not None or self.device == "cpu":
            return
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._decode_step_eager()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._decode_step_eager()
        self._graph = g
        log.info("T3 decode step captured as CUDA graph (%s)", self.t3_dtype)

    def _decode_step(self, tok: torch.Tensor, pos: int) -> torch.Tensor:
        self._s_tok.copy_(tok)
        self._s_pos.fill_(pos)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._decode_step_eager()
        return self._s_logits.float()

    def _cond_for_dtype(self, t3_cond):
        """T3 runs in fp16 but conditionals are stored fp32; cast once per voice and memoize on the object."""
        cached = getattr(t3_cond, "_cast_cache", None)
        if cached is not None and cached[0] == self.t3_dtype:
            return cached[1]
        import copy

        c = copy.copy(t3_cond).to(device=self.device, dtype=self.t3_dtype)
        try:
            t3_cond._cast_cache = (self.t3_dtype, c)
        except Exception:  # noqa: BLE001 - frozen dataclass or similar
            pass
        return c

    # -------------------------------------------------------------- tokens
    @torch.inference_mode()
    def _stream_tokens(self, text: str, t3_cond, params: GenParams) -> Iterator[torch.Tensor]:
        """Yield 1-D LongTensors of new speech tokens (block by block)."""
        from chatterbox.tts_turbo import punc_norm

        t3 = self.model.t3
        hp = t3.hp
        text = punc_norm(text)
        text_tokens = self.model.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        procs = LogitsProcessorList()
        if params.temperature > 0 and params.temperature != 1.0:
            procs.append(TemperatureLogitsWarper(params.temperature))
        procs.append(TopKLogitsWarper(1000))
        if params.top_p < 1.0:
            procs.append(TopPLogitsWarper(params.top_p))
        if params.repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))

        start = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        t3_cond = self._cond_for_dtype(t3_cond)
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
        L = embeds.shape[1]
        max_new = min(self.cfg.max_gen_len, self.cfg.max_cache_len - L - 1)
        if max_new <= 0:
            raise ValueError("input too long for the T3 cache; shorten the text or reference clip")

        self._prefill(embeds)
        self._capture()
        logits = self._s_logits.float()

        buf = torch.empty((1, max_new + 1), dtype=torch.long, device=self.device)
        tok = torch.multinomial(F.softmax(procs(start, logits), dim=-1), 1)
        buf[:, 0] = tok[:, 0]
        n = 1

        block_sizes = iter([self.cfg.first_block, *self.cfg.block_growth])
        target = next(block_sizes)
        emitted = 0
        stop = hp.stop_speech_token

        for i in range(max_new):
            logits = self._decode_step(tok, L + i)
            logits = procs(buf[:, :n], logits)
            tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            # one host sync per token: unavoidable, we need to know when to stop
            if tok.item() == stop:
                break
            buf[:, n] = tok[:, 0]
            n += 1
            if n - emitted >= target:
                yield buf[0, emitted:n].clone()
                emitted = n
                target = next(block_sizes, self.cfg.max_block)

        if n > emitted:
            yield buf[0, emitted:n].clone()

    # -------------------------------------------------------------- vocoder
    @torch.inference_mode()
    def _flow_mels(self, tokens: torch.Tensor, ref: dict, noise: torch.Tensor) -> torch.Tensor:
        """`flow.inference` re-implemented: same encoder path, then the CFM solve
        through the bucketed CUDA graphs, with one fixed noise tensor for the
        whole request (prompt region included) so recomputed prefixes match."""
        from chatterbox.models.s3gen.utils.mask import make_pad_mask

        flow = self.flow
        emb = F.normalize(torch.atleast_2d(ref["embedding"]), dim=1)
        emb = flow.spk_embed_affine_layer(emb)

        tok = torch.cat([ref["prompt_token"], tokens], dim=1)
        tok_len = ref["prompt_token_len"] + tokens.shape[1]
        mask = (~make_pad_mask(tok_len)).unsqueeze(-1).to(emb)
        h, _ = flow.encoder(flow.input_embedding(tok.long()) * mask, tok_len)
        h = flow.encoder_proj(h)  # (1, T, 80)
        T = h.shape[1]
        mel_len1 = ref["prompt_feat"].shape[1]

        cond = torch.zeros(1, T, 80, device=self.device, dtype=h.dtype)
        cond[:, :mel_len1] = ref["prompt_feat"]
        cond = cond.transpose(1, 2).contiguous()
        mu = h.transpose(1, 2).contiguous()
        mask = torch.ones(1, 1, T, device=self.device, dtype=h.dtype)
        z = noise[..., :T].to(h.dtype)

        if self.cfm is not None:
            feat = self.cfm.solve(z, mu, mask, emb, cond)
        else:
            feat = self._euler_eager(z, mu, mask, emb, cond)
        return feat[:, :, mel_len1:]

    def _euler_eager(self, x, mu, mask, spks, cond):
        est = self.flow.decoder.estimator
        t_span = torch.linspace(0, 1, 3, device=self.device)
        for t, r in zip(t_span[:-1], t_span[1:]):
            t, r = t[None], r[None]
            x = x + (r - t) * est(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        return x

    @torch.inference_mode()
    def _vocode(self, tokens: torch.Tensor, ref: dict, noise: torch.Tensor, finalize: bool, state: dict) -> torch.Tensor | None:
        """Flow-match the full token sequence, vocode only the new mel frames."""
        tokens = tokens[tokens < SPEECH_VOCAB]
        if finalize:
            tokens = torch.cat([tokens, torch.full((3,), S3GEN_SIL, dtype=tokens.dtype, device=self.device)])
        tokens = tokens.unsqueeze(0)
        n_tok = tokens.shape[1]
        n_mel_full = TOKEN_MEL_RATIO * n_tok
        # frames without enough right-context are held back until the next block
        # (the library's own finalize=False path trims h but not its mask, so
        # we run finalize=True and trim ourselves - same encoder pass either way)
        n_mel = n_mel_full if finalize else n_mel_full - TOKEN_MEL_RATIO * self.flow.pre_lookahead_len
        if n_mel <= state["emitted"]:
            return None

        mels = self._flow_mels(tokens, ref, noise)
        mels = mels[:, :, :n_mel].to(self.model.s3gen.dtype)
        new = mels[:, :, state["emitted"] :]
        state["emitted"] = mels.shape[2]

        cache = state.get("hift")
        if cache is not None:
            new = torch.cat([cache["mel"], new], dim=2)
            source = cache["source"]
        else:
            source = torch.zeros(1, 1, 0, device=self.device)
        speech, source = self.hift.inference(speech_feat=new, cache_source=source)
        if cache is not None:
            L = SOURCE_CACHE
            speech[:, :L] = speech[:, :L] * self.window[L:] + cache["speech"][:, -L:] * self.window[:L]
        else:
            speech[:, : len(self.trim_fade)] *= self.trim_fade
        if not finalize:
            state["hift"] = {
                "mel": new[:, :, -MEL_CACHE_FRAMES:],
                "source": source[:, :, -SOURCE_CACHE:],
                "speech": speech[:, -SOURCE_CACHE:],
            }
            speech = speech[:, :-SOURCE_CACHE]
        return speech

    # -------------------------------------------------------------- public
    def stream(self, text: str, conds, params: GenParams) -> Iterator[np.ndarray]:
        """Yield float32 24 kHz audio blocks for `text` as soon as each is ready."""
        if params.seed is not None:
            torch.manual_seed(params.seed)
        ref = conds.gen
        noise = torch.randn(1, 80, MAX_MEL_FRAMES, device=self.device, dtype=self.model.s3gen.dtype)
        state: dict = {"emitted": 0}
        all_tokens: list[torch.Tensor] = []

        # every block is vocoded as soon as it exists (flow drops a 3-token
        # lookahead so the boundary is stable); a final pass with finalize=True
        # emits those held-back frames plus the closing silence
        for block in self._stream_tokens(text, conds.t3, params):
            all_tokens.append(block)
            out = self._vocode(torch.cat(all_tokens), ref, noise, finalize=False, state=state)
            if out is not None:
                yield self._finish(out)

        if all_tokens:
            out = self._vocode(torch.cat(all_tokens), ref, noise, finalize=True, state=state)
            if out is not None:
                yield self._finish(out)

    def _finish(self, speech: torch.Tensor) -> np.ndarray:
        wav = speech.squeeze(0).float().cpu().numpy()
        try:
            wav = self.model.watermarker.apply_watermark(wav, sample_rate=self.model.sr)
        except Exception:  # noqa: BLE001 - very short blocks can trip the watermarker
            pass
        return wav.astype(np.float32)
