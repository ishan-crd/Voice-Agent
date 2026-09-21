"""Streaming audio encoders.

Every encoder takes float32 mono chunks at the engine sample rate and yields
bytes that can be written straight into a chunked HTTP response.

  pcm    raw signed 16-bit little-endian (lowest latency; what voice agents want)
  wav    pcm with a streaming RIFF header (unknown length)
  mulaw  8-bit G.711 mu-law, default 8 kHz (Twilio / SIP media streams)
  alaw   8-bit G.711 A-law
  mp3 / opus / aac / flac   via an ffmpeg subprocess, encoded as chunks arrive
"""
from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger("tts.audio")

FORMATS = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mulaw": "audio/basic",
    "alaw": "audio/x-alaw-basic",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}
FFMPEG_FORMATS = {"mp3", "opus", "aac", "flac"}


def _find_ffmpeg() -> str | None:
    """PATH first, then TTS_FFMPEG, then the winget/scoop/choco spots on Windows
    (a process started before the install won't see the updated PATH)."""
    import glob
    import os

    for cand in (os.environ.get("TTS_FFMPEG"), shutil.which("ffmpeg")):
        if cand and Path(cand).exists():
            return cand
    local = os.environ.get("LOCALAPPDATA", "")
    patterns = [
        rf"{local}\Microsoft\WinGet\Links\ffmpeg.exe",
        rf"{local}\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
        rf"{os.environ.get('USERPROFILE', '')}\scoop\shims\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


FFMPEG = _find_ffmpeg()


def content_type(fmt: str) -> str:
    return FORMATS[fmt]


# ------------------------------------------------------------------ helpers
def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    return AF.resample(t, sr_in, sr_out).squeeze(0).numpy()


def time_stretch(x: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change tempo without changing pitch (OpenAI `speed` semantics)."""
    if abs(speed - 1.0) < 1e-3:
        return x
    import librosa

    return librosa.effects.time_stretch(x, rate=speed).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def pcm16_bytes(x: np.ndarray) -> bytes:
    return to_int16(x).tobytes()


def wav_header(sr: int, channels: int = 1, bits: int = 16) -> bytes:
    """RIFF header with 'unknown' sizes so it can be streamed."""
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits),
            b"data",
            struct.pack("<I", 0xFFFFFFFF),
        ]
    )


# --------------------------------------------------------------- G.711
_MU = 255.0
_BIAS = 0x84
_CLIP = 32635


def mulaw_encode(x: np.ndarray) -> bytes:
    """Float32 [-1,1] -> 8-bit mu-law (ITU-T G.711)."""
    pcm = to_int16(x).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0).astype(np.int32)
    mag = np.minimum(np.abs(pcm), _CLIP) + _BIAS
    exponent = np.floor(np.log2(mag)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    out = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return out.astype(np.uint8).tobytes()


def alaw_encode(x: np.ndarray) -> bytes:
    """Float32 [-1,1] -> 8-bit A-law (ITU-T G.711)."""
    pcm = to_int16(x).astype(np.int32)
    sign = np.where(pcm >= 0, 0x80, 0).astype(np.int32)
    mag = np.minimum(np.abs(pcm), 32767) >> 3  # 13-bit
    exponent = np.zeros_like(mag)
    nz = mag > 0x1F
    exponent[nz] = np.floor(np.log2(mag[nz])).astype(np.int32) - 4
    exponent = np.clip(exponent, 0, 7)
    mantissa = np.where(exponent == 0, mag >> 1, (mag >> exponent) & 0x0F)
    out = (sign | (exponent << 4) | mantissa) ^ 0x55
    return out.astype(np.uint8).tobytes()


# ---------------------------------------------------------- ffmpeg pipe
class FfmpegEncoder:
    """Feed PCM in, read encoded bytes out, without waiting for the end.

    ffmpeg buffers a little internally, so the first encoded bytes trail the
    first PCM by ~30-60 ms for mp3/opus.  Use `pcm` when latency matters.
    """

    _ARGS = {
        "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "64k", "-write_xing", "0"],
        "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k", "-application", "voip", "-frame_duration", "20"],
        "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "64k"],
        "flac": ["-f", "flac", "-c:a", "flac"],
    }

    def __init__(self, fmt: str, sr: int) -> None:
        if FFMPEG is None:
            raise RuntimeError("ffmpeg not found on PATH; use response_format=pcm or wav")
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-flush_packets", "1", *self._ARGS[fmt], "pipe:1",
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._chunks: "list[bytes | None]" = []
        self._cv = threading.Condition()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        while True:
            data = self.proc.stdout.read1(65536)  # type: ignore[attr-defined]
            with self._cv:
                self._chunks.append(data if data else None)
                self._cv.notify_all()
            if not data:
                return

    def write(self, pcm: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            err = self.proc.stderr.read().decode(errors="ignore") if self.proc.stderr else ""
            raise RuntimeError(f"ffmpeg died: {err.strip()}")

    def drain(self) -> Iterator[bytes]:
        """Yield whatever has been encoded so far (non-blocking)."""
        with self._cv:
            out, self._chunks = self._chunks, []
        for c in out:
            if c:
                yield c

    def finish(self) -> Iterator[bytes]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        while True:
            with self._cv:
                while not self._chunks:
                    self._cv.wait(timeout=5.0)
                    if self.proc.poll() is not None and not self._chunks:
                        return
                out, self._chunks = self._chunks, []
            for c in out:
                if c is None:
                    self.proc.wait(timeout=5)
                    return
                yield c

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


# --------------------------------------------------------------- facade
class StreamEncoder:
    """Stateful per-response encoder: `encode(chunk)` then `finish()`."""

    def __init__(self, fmt: str, sr_in: int, sample_rate: int | None = None, speed: float = 1.0) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"unsupported response_format {fmt!r}; choose from {sorted(FORMATS)}")
        self.fmt = fmt
        self.sr_in = sr_in
        self.speed = speed
        if fmt in ("mulaw", "alaw"):
            self.sr_out = sample_rate or 8000
        else:
            self.sr_out = sample_rate or sr_in
        self._ff: FfmpegEncoder | None = FfmpegEncoder(fmt, self.sr_out) if fmt in FFMPEG_FORMATS else None
        self._header_sent = False

    @property
    def content_type(self) -> str:
        return FORMATS[self.fmt]

    def encode(self, audio: np.ndarray) -> Iterator[bytes]:
        audio = time_stretch(audio, self.sr_in, self.speed)
        audio = resample(audio, self.sr_in, self.sr_out)
        if self.fmt == "pcm":
            yield pcm16_bytes(audio)
        elif self.fmt == "wav":
            if not self._header_sent:
                self._header_sent = True
                yield wav_header(self.sr_out)
            yield pcm16_bytes(audio)
        elif self.fmt == "mulaw":
            yield mulaw_encode(audio)
        elif self.fmt == "alaw":
            yield alaw_encode(audio)
        else:
            assert self._ff is not None
            self._ff.write(pcm16_bytes(audio))
            yield from self._ff.drain()

    def finish(self) -> Iterator[bytes]:
        if self._ff is not None:
            yield from self._ff.finish()

    def close(self) -> None:
        if self._ff is not None:
            self._ff.close()
