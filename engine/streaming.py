"""Token-level streaming for Chatterbox Turbo and Multilingual.

The stock `generate()` runs the T3 decoder to completion, then vocodes the
whole token sequence.  Here the T3 loop yields speech tokens in growing
blocks and each block is vocoded immediately (CosyVoice2-style chunked
flow-matching + HiFT with a source cache and a short crossfade), so the first
audio leaves the GPU after ~12 tokens instead of after the whole sentence.

Both hot loops are captured as CUDA graphs:
  * one T3 decode step (StaticCache, fp16)              ~4 ms/token
  * the whole CFM Euler solve, per mel-length bucket    ~30 ms (Turbo, 2 steps)

Turbo   : GPT2 backbone, no CFG, mean-flow vocoder (2 steps)
Multi   : Llama backbone, CFG (batch 2), classic CFM vocoder (10 steps, CFG)
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinPLogitsWarper,
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


# ============================================================ CFM graphs
class CfmGraphs:
    """The CFM Euler solve captured as one CUDA graph per mel-length bucket.

    The estimator (U-Net + transformer blocks) is launch-bound: ~135 ms per
    call at any length in eager mode, a few ms replayed.  Inputs are padded
    with a zero mask to the bucket length; padded frames are masked out of
    attention and multiplied away in the conv blocks, so valid frames match
    the unpadded result to ~1e-3.

    meanflow=True : 2 steps, linear t, no CFG (Turbo)
    meanflow=False: n steps, cosine t, CFG with `inference_cfg_rate` (Multilingual)
    """

    def __init__(self, cfm, device, bucket: int, max_frames: int, *, meanflow: bool, n_steps: int) -> None:
        self.cfm = cfm
        self.est = cfm.estimator
        self.device = device
        self.bucket = bucket
        self.max_frames = max_frames
        self.meanflow = meanflow
        self.n_steps = n_steps
        self.cfg_rate = float(getattr(cfm, "inference_cfg_rate", 0.0))
        t = torch.linspace(0, 1, n_steps + 1, device=device)
        if not meanflow and getattr(cfm, "t_scheduler", "") == "cosine":
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        self.t_span = t
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, dict[str, torch.Tensor]]] = {}
        self._pool = None

    def euler(self, x, mu, mask, spks, cond) -> torch.Tensor:
        if self.meanflow:
            for t, r in zip(self.t_span[:-1], self.t_span[1:]):
                t, r = t[None], r[None]
                dxdt = self.est(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
                x = x + (r - t) * dxdt
            return x
        # classic CFM with classifier-free guidance: batch 2 = (cond, uncond)
        zeros_mu, zeros_spk, zeros_cond = torch.zeros_like(mu), torch.zeros_like(spks), torch.zeros_like(cond)
        mask_in = torch.cat([mask, mask])
        mu_in = torch.cat([mu, zeros_mu])
        spks_in = torch.cat([spks, zeros_spk])
        cond_in = torch.cat([cond, zeros_cond])
        for t, r in zip(self.t_span[:-1], self.t_span[1:]):
            t_in = t.expand(2)
            dxdt = self.est(torch.cat([x, x]), mask=mask_in, mu=mu_in, t=t_in, spks=spks_in, cond=cond_in, r=None)
            d_cond, d_uncond = dxdt[:1], dxdt[1:]
            dxdt = (1.0 + self.cfg_rate) * d_cond - self.cfg_rate * d_uncond
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
        t0 = time.perf_counter()
        B = self._bucket_for(min_frames) or self.bucket
        while B <= self.max_frames:
            self.capture(B, spks_dim)
            B += self.bucket
        log.info(
            "captured %d CFM graphs (%d..%d frames, %d steps%s) in %.1fs",
            len(self.graphs), min(self.graphs), max(self.graphs), self.n_steps,
            "" if self.meanflow else f", cfg {self.cfg_rate}", time.perf_counter() - t0,
        )

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


# ========================================================= shared base
class ChunkedStreamer:
    """Shared machinery: fp16 T3 + static cache + graph decode step, chunked
    vocoder.  Subclasses implement `_prepare(text, t3_cond, params)` -> prefill
    embeds and `_step_embed(tok, i)` -> next input embeds, plus `_mix_logits`."""

    cfg_batch = 1  # 2 when the T3 runs classifier-free guidance
    cfm_meanflow = True
    cfm_steps = 2

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
        self._s_step = torch.zeros((1,), dtype=torch.long, device=self.device)  # generated-token index
        self._s_logits = torch.empty((self.cfg_batch, t3.hp.speech_tokens_dict_size), device=self.device, dtype=self.t3_dtype)

        self.cfm: CfmGraphs | None = None
        if self.cfg.cuda_graph and self.device != "cpu":
            import chatterbox.models.s3gen.decoder as _dec

            _dec.add_optional_chunk_mask = _chunk_mask_capture_safe
            self.cfm = CfmGraphs(
                self.flow.decoder, self.device, self.cfg.cfm_bucket, self.cfg.cfm_max_frames,
                meanflow=self.cfm_meanflow, n_steps=self.cfm_steps,
            )

    def warmup_graphs(self, prompt_frames: int) -> None:
        """Capture every CFM bucket a stream can hit; call once at startup."""
        if self.cfm is not None and not self.cfm.graphs:
            spks_dim = self.flow.spk_embed_affine_layer.out_features
            self.cfm.capture_all(prompt_frames + TOKEN_MEL_RATIO * self.cfg.first_block, spks_dim)

    # ------------------------------------------------------------ subclass API
    def _prepare(self, text: str, t3_cond, params: GenParams) -> torch.Tensor:
        raise NotImplementedError

    def _step_embed(self) -> torch.Tensor:
        """Input embeds for the decode step, built from the static tok/step buffers."""
        raise NotImplementedError

    def _mix_logits(self, logits: torch.Tensor, params: GenParams) -> torch.Tensor:
        return logits

    def _processors(self, params: GenParams) -> LogitsProcessorList:
        raise NotImplementedError

    def _max_new(self, text: str) -> int:
        return self.cfg.max_gen_len

    # ------------------------------------------------------- T3 decode step
    def _decode_step_eager(self) -> None:
        t3 = self.model.t3
        out = t3.tfmr(inputs_embeds=self._step_embed(), past_key_values=self._cache, use_cache=True, cache_position=self._s_pos)
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
        log.info("%s: T3 decode step captured as CUDA graph (%s)", type(self).__name__, self.t3_dtype)

    def _decode_step(self, tok: torch.Tensor, pos: int, step: int) -> torch.Tensor:
        self._s_tok.copy_(tok)
        self._s_pos.fill_(pos)
        self._s_step.fill_(step)
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
        c = copy.copy(t3_cond).to(device=self.device, dtype=self.t3_dtype)
        try:
            t3_cond._cast_cache = (self.t3_dtype, c)
        except Exception:  # noqa: BLE001
            pass
        return c

    # -------------------------------------------------------------- tokens
    @torch.inference_mode()
    def _stream_tokens(self, text: str, t3_cond, params: GenParams) -> Iterator[torch.Tensor]:
        """Yield 1-D LongTensors of new speech tokens (block by block)."""
        hp = self.model.t3.hp
        embeds = self._prepare(text, self._cond_for_dtype(t3_cond), params)
        L = embeds.shape[1]
        max_new = min(self._max_new(text), self.cfg.max_cache_len - L - 1)
        if max_new <= 0:
            raise ValueError("input too long for the T3 cache; shorten the text or reference clip")
        procs = self._processors(params)

        self._prefill(embeds)
        self._capture()

        buf = torch.empty((1, max_new + 1), dtype=torch.long, device=self.device)
        start = torch.full((1, 1), hp.start_speech_token, dtype=torch.long, device=self.device)
        logits = self._mix_logits(self._s_logits.float(), params)
        tok = torch.multinomial(F.softmax(procs(start, logits), dim=-1), 1)
        buf[:, 0] = tok[:, 0]
        n = 1

        block_sizes = iter([self.cfg.first_block, *self.cfg.block_growth])
        target = next(block_sizes)
        emitted = 0
        stop = hp.stop_speech_token

        for i in range(max_new):
            logits = self._mix_logits(self._decode_step(tok, L + i, i), params)
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
            feat = CfmGraphs(self.flow.decoder, self.device, 1, 0, meanflow=self.cfm_meanflow, n_steps=self.cfm_steps).euler(z, mu, mask, emb, cond)
        return feat[:, :, mel_len1:]

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
        # we run the full encoder and trim ourselves - same pass either way)
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
    def stream(self, text: str, conds, params: GenParams, language: str = "en") -> Iterator[np.ndarray]:
        """Yield float32 24 kHz audio blocks for `text` as soon as each is ready."""
        if params.seed is not None:
            torch.manual_seed(params.seed)
        self.language = language
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


# ================================================================ Turbo
class TurboStreamer(ChunkedStreamer):
    cfg_batch = 1
    cfm_meanflow = True
    cfm_steps = 2

    def _prepare(self, text: str, t3_cond, params: GenParams) -> torch.Tensor:
        from chatterbox.tts_turbo import punc_norm

        t3 = self.model.t3
        text_tokens = self.model.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(self.device)
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
        return embeds

    def _step_embed(self) -> torch.Tensor:
        return self.model.t3.speech_emb(self._s_tok)

    def _processors(self, params: GenParams) -> LogitsProcessorList:
        procs = LogitsProcessorList()
        if params.temperature > 0 and params.temperature != 1.0:
            procs.append(TemperatureLogitsWarper(params.temperature))
        procs.append(TopKLogitsWarper(1000))
        if params.top_p < 1.0:
            procs.append(TopPLogitsWarper(params.top_p))
        if params.repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))
        return procs


# ========================================================= Multilingual
class MultilingualStreamer(ChunkedStreamer):
    """Llama T3 with classifier-free guidance (batch 2: conditional / text-free)
    and learned speech position embeddings; classic 10-step CFG vocoder.

    The stock path also runs an attention-based "alignment stream analyzer"
    that forces EOS on runaway generations; it needs eager attention with
    attention outputs and cannot be graph-captured, so it is replaced with a
    text-length-derived cap on generated tokens.
    """

    cfg_batch = 2
    cfm_meanflow = False
    cfm_steps = 10

    def _prepare(self, text: str, t3_cond, params: GenParams) -> torch.Tensor:
        from chatterbox.mtl_tts import punc_norm

        t3 = self.model.t3
        hp = t3.hp
        lang = (self.language or "en").lower()
        text_tokens = self.model.tokenizer.text_to_tokens(punc_norm(text), language_id=lang).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # (cond, uncond)
        text_tokens = F.pad(text_tokens, (1, 0), value=hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=hp.stop_text_token)

        # exaggeration lives in the conditionals; swap it in without mutating the cached object
        if float(t3_cond.emotion_adv.flatten()[0]) != float(params.exaggeration):
            t3_cond = copy.copy(t3_cond)
            t3_cond.emotion_adv = torch.full((1, 1, 1), params.exaggeration, device=self.device, dtype=self.t3_dtype)

        start = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=params.cfg_weight)
        # the reference loop appends a second BOS with fixed position 0; keep the model's contract
        bos = t3.speech_emb(start[:1]) + t3.speech_pos_emb.get_fixed_embedding(0)
        embeds = torch.cat([embeds, torch.cat([bos, bos])], dim=1)
        self._cfg_weight = params.cfg_weight
        return embeds

    def _step_embed(self) -> torch.Tensor:
        t3 = self.model.t3
        e = t3.speech_emb(self._s_tok) + t3.speech_pos_emb.get_fixed_embedding(self._s_step + 1)
        return torch.cat([e, e])

    def _mix_logits(self, logits: torch.Tensor, params: GenParams) -> torch.Tensor:
        cond, uncond = logits[0:1], logits[1:2]
        return cond + params.cfg_weight * (cond - uncond)

    def _processors(self, params: GenParams) -> LogitsProcessorList:
        procs = LogitsProcessorList()
        if params.repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))
        if params.temperature > 0 and params.temperature != 1.0:
            procs.append(TemperatureLogitsWarper(params.temperature))
        procs.append(MinPLogitsWarper(min_p=0.05))
        if params.top_p < 1.0:
            procs.append(TopPLogitsWarper(params.top_p))
        return procs

    def _max_new(self, text: str) -> int:
        # replaces the alignment analyzer's runaway guard: ~8 chars/s of speech, generous
        return min(self.cfg.max_gen_len, int(25 * (1.5 + 0.15 * len(text))))
