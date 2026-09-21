"""Speech-to-text with Whisper (transformers, fp16 on the GPU).

Used by the Talk feature.  Runs in its own thread, not on the TTS worker, so
listening never blocks speaking.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("tts.stt")

_LANG_RE = re.compile(r"<\|([a-z]{2,3})\|>")


@dataclass
class Transcript:
    text: str
    language: str
    ms: float


class Whisper:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cuda") -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        t0 = time.perf_counter()
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self._lock = threading.Lock()
        self.model_id = model_id
        # warm the kernels once
        self.transcribe(np.zeros(16_000, dtype=np.float32))
        log.info("whisper %s ready in %.1fs", model_id, time.perf_counter() - t0)

    def transcribe(self, audio16k: np.ndarray, language: str | None = None) -> Transcript:
        """audio16k: float32 mono at 16 kHz. `language` None = auto-detect."""
        import torch

        t0 = time.perf_counter()
        feats = self.processor(audio16k, sampling_rate=16_000, return_tensors="pt").input_features.to(self.device, self.dtype)
        kw = {"task": "transcribe", "max_new_tokens": 160}
        if language:
            kw["language"] = language
        t1 = time.perf_counter()
        with self._lock, torch.inference_mode():
            ids = self.model.generate(feats, **kw)
            torch.cuda.synchronize()
        log.debug("whisper: features %.0f ms, generate %.0f ms (%d tok, lang=%s)", (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, ids.shape[1], language)
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        m = _LANG_RE.search(raw)
        lang = language or (m.group(1) if m else "en")
        if lang == "en" and re.search(r"[ऀ-ॿ]", text):
            lang = "hi"  # the language token is not always echoed back; trust the script
        return Transcript(text=text, language=lang, ms=(time.perf_counter() - t0) * 1000)
