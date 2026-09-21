"""Sentence-boundary chunking tuned for streaming TTS.

Time-to-first-audio is dominated by how much text the first `generate()` call
has to chew, so the first chunk is deliberately short (a clause, ~10 words).
Everything after that is whole sentences, merged when tiny and split at
commas when too long for the model to stay stable.
"""
from __future__ import annotations

import re

from .config import settings

# Common abbreviations that end with a period but do not end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
    "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "no", "approx", "dept", "est", "fig",
}

# Sentence terminators for Latin scripts plus Devanagari danda (। ॥) and CJK.
_TERMINATORS = ".!?।॥。！？"
_SENT_RE = re.compile(rf"(.*?[{re.escape(_TERMINATORS)}]+[\"'”’)\]]*)(\s+|$)", re.S)
_CLAUSE_RE = re.compile(r"[,;:—–\-]\s+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    # newlines usually mean a hard break in agent output; treat as sentence end
    # unless the previous sentence is already terminated
    text = re.sub(rf"([{re.escape(_TERMINATORS)}][\"'”’)\]]*)\s*\n+", r"\1 ", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = text.replace("\n", " ")
    return _WS_RE.sub(" ", text).strip()


def _is_abbrev_end(sentence: str) -> bool:
    tail = sentence.rstrip(".!?\"'”’)]").split()
    if not tail:
        return False
    last = tail[-1].lower().strip("(\"'")
    if last in _ABBREV:
        return True
    # single capital letter initial: "J. K. Rowling"
    if len(last) == 1 and last.isalpha():
        return True
    # decimal / version numbers: "3.14" - only if terminated by '.' and next is a digit, handled by caller
    return False


def split_sentences(text: str) -> list[str]:
    """Split on sentence terminators with abbreviation and decimal guards."""
    text = normalize(text)
    if not text:
        return []
    out: list[str] = []
    pos = 0
    buf = ""
    for m in _SENT_RE.finditer(text):
        piece = m.group(1)
        pos = m.end()
        buf = f"{buf} {piece}".strip() if buf else piece
        nxt = text[pos : pos + 1]
        # "3.14", "v2.0": digit on both sides of a period
        if piece.endswith(".") and nxt.isdigit() and len(piece) > 1 and piece[-2].isdigit():
            continue
        if _is_abbrev_end(piece) and nxt and not nxt.isupper():
            continue
        if _is_abbrev_end(piece) and piece.lower().rstrip(".") .split()[-1] in _ABBREV:
            # "Dr. Smith" - capital after abbreviation is still not a boundary
            continue
        out.append(buf)
        buf = ""
    rest = text[pos:].strip()
    if buf or rest:
        out.append(f"{buf} {rest}".strip())
    return [s for s in out if s]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    cur = ""
    for clause in _CLAUSE_RE.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if cur and len(cur) + len(clause) + 2 > max_chars:
            parts.append(cur)
            cur = clause
        else:
            cur = f"{cur}, {clause}" if cur else clause
    if cur:
        parts.append(cur)
    # a single clause longer than max_chars: hard-split on words
    final: list[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            final.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            final.append(p)
    return final


def _first_chunk(sentence: str, max_words: int) -> tuple[str, str]:
    """Cut a short lead-in off the first sentence at a clause boundary if possible."""
    words = sentence.split()
    if len(words) <= max_words + 3:  # not worth splitting a short sentence
        return sentence, ""
    m = None
    for cand in _CLAUSE_RE.finditer(sentence):
        n_words = len(sentence[: cand.start()].split())
        if 3 <= n_words <= max_words + 4:
            m = cand
        elif n_words > max_words + 4:
            break
    if m:
        return sentence[: m.start()].strip() + ",", sentence[m.end() :].strip()
    head = " ".join(words[:max_words])
    tail = " ".join(words[max_words:])
    return head + ",", tail


def chunk_text(
    text: str,
    *,
    first_chunk_words: int | None = None,
    max_chars: int | None = None,
    min_chars: int = 20,
) -> list[str]:
    """Turn free text into ordered chunks for sequential synthesis."""
    first_chunk_words = first_chunk_words or settings.first_chunk_words
    max_chars = max_chars or settings.max_chunk_chars

    sentences = split_sentences(text)
    if not sentences:
        return []

    # merge tiny fragments ("Hi.", "Okay.") into their neighbour
    merged: list[str] = []
    for s in sentences:
        if merged and (len(merged[-1]) < min_chars or len(s) < min_chars) and len(merged[-1]) + len(s) < max_chars:
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)

    chunks: list[str] = []
    for s in merged:
        chunks.extend(_split_long(s, max_chars))

    if first_chunk_words > 0 and chunks:
        head, tail = _first_chunk(chunks[0], first_chunk_words)
        chunks[0] = head
        if tail:
            chunks.insert(1, tail)
    return chunks
