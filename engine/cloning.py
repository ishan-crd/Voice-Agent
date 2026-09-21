"""Pro cloning from long recordings.

Chatterbox is zero-shot: it conditions on the first ~10 s of the reference
clip and nothing else.  Given a 1-10 minute recording we can still do
better than "use the first ten seconds":

  1. find every clean, fully-voiced 8 s window (energy VAD, no clipping,
     stable level)
  2. compute the speaker identity vectors on *all* of them and average
     (T3's voice-encoder embedding and S3Gen's x-vector both average well)
  3. use the window whose identity is closest to that average as the
     acoustic prompt (prompt tokens + mel for the vocoder)

Identity becomes stable instead of depending on which 10 s happened to be
first; the prompt is the most "typical" stretch of the speaker.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("tts.cloning")

SR = 24_000
WIN_S = 8.0
HOP_S = 2.0
MAX_WINDOWS = 16
MIN_WINDOWS = 1


@dataclass
class Window:
    start: int
    end: int
    score: float
    audio: np.ndarray


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * frame].reshape(n, frame)
    return np.sqrt((f**2).mean(axis=1) + 1e-12)


def find_windows(x: np.ndarray, sr: int = SR) -> list[Window]:
    """Candidate 8 s windows ranked by voiced ratio, level stability and no clipping."""
    frame = int(0.03 * sr)
    rms = _frame_rms(x, frame)
    if len(rms) == 0:
        return []
    db = 20 * np.log10(rms + 1e-9)
    noise_floor = np.percentile(db, 10)
    voiced = db > max(noise_floor + 12, -45)  # 12 dB above the floor counts as speech

    win = int(WIN_S * sr)
    hop = int(HOP_S * sr)
    fpw = win // frame
    out: list[Window] = []
    for start in range(0, max(1, len(x) - win + 1), hop):
        s_f = start // frame
        seg_v = voiced[s_f : s_f + fpw]
        if len(seg_v) < fpw * 0.9:
            break
        voiced_ratio = float(seg_v.mean())
        seg = x[start : start + win]
        clip = float((np.abs(seg) > 0.99).mean())
        level = db[s_f : s_f + fpw][seg_v]
        stability = float(np.std(level)) if len(level) > 2 else 99.0
        # longest silent run inside the window (pauses > 1.2 s make a poor prompt)
        longest_gap = 0
        run = 0
        for v in seg_v:
            run = 0 if v else run + 1
            longest_gap = max(longest_gap, run)
        gap_s = longest_gap * frame / sr
        score = voiced_ratio - 5 * clip - 0.01 * stability - (0.5 if gap_s > 1.2 else 0.0)
        if voiced_ratio >= 0.55 and clip < 0.01:
            out.append(Window(start, start + win, score, seg))
    out.sort(key=lambda w: w.score, reverse=True)
    # keep the best, but spread across the recording so one loud minute doesn't dominate
    chosen: list[Window] = []
    for w in out:
        if all(abs(w.start - c.start) >= win // 2 for c in chosen):
            chosen.append(w)
        if len(chosen) >= MAX_WINDOWS:
            break
    return chosen


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    import librosa

    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x.astype(np.float32)


def prepare_pro(engine, kind: str, path: Path, exaggeration: float = 0.5):
    """GPU-thread only. Returns Conditionals for `kind` built from a long recording."""
    import torch

    model = engine.models[kind]
    x = load_mono(path)
    dur = len(x) / SR
    windows = find_windows(x)
    if len(windows) < MIN_WINDOWS:
        raise ValueError(f"could not find a clean 8 s stretch of speech in the recording ({dur:.0f} s)")
    log.info("pro clone %s: %.0f s of audio, %d candidate windows", kind, dur, len(windows))

    # identity vectors over every window
    import librosa

    segs16 = [librosa.resample(w.audio, orig_sr=SR, target_sr=16_000) for w in windows]
    ve = torch.from_numpy(model.ve.embeds_from_wavs(segs16, sample_rate=16_000))  # (N, 256)
    spk = model.s3gen.speaker_encoder
    with torch.inference_mode():
        xv = torch.cat(
            [spk.inference(torch.from_numpy(s).unsqueeze(0).to(device=model.device, dtype=model.s3gen.dtype)).detach().float().cpu() for s in segs16]
        )  # (N, D)
    ve_mean = ve.mean(dim=0, keepdim=True)
    xv_mean = xv.mean(dim=0, keepdim=True)

    # the most typical window = closest to the mean identity (cosine)
    sim = torch.nn.functional.cosine_similarity(ve, ve_mean.expand_as(ve), dim=1)
    # only the cleaner half of the windows are eligible to be the prompt
    scores = torch.tensor([w.score for w in windows])
    eligible = scores >= scores.median()
    best = int(torch.where(eligible, sim, torch.full_like(sim, -1.0)).argmax())
    log.info("pro clone %s: prompt window at %.1f s (similarity %.3f, score %.2f)", kind, windows[best].start / SR, float(sim[best]), windows[best].score)

    # run the model's own conditioning on the best window, then swap in the averaged identity
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        sf.write(tmp, windows[best].audio, SR)
        conds = engine.prepare_conditionals(kind, str(tmp), exaggeration=exaggeration)
    finally:
        tmp.unlink(missing_ok=True)

    dev = model.device
    conds.t3.speaker_emb = ve_mean.to(device=dev, dtype=conds.t3.speaker_emb.dtype)
    conds.gen["embedding"] = xv_mean.to(device=dev, dtype=conds.gen["embedding"].dtype)
    return conds
