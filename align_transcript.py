#!/usr/bin/env python3
"""
align_transcript.py — Align a clean source transcript to an audio file and emit
timed subtitles (.srt) suitable for importing into LingQ.

Why this exists
---------------
Podcast source transcripts are clean but have NO timestamps, and the audio often
contains material the transcript omits (ads, station promos, "Wait Wait..."
spots, etc.). Naive forced aligners assume the text and audio contain the same
words and smear the text across the ad regions.

Approach
--------
1. Transcribe the audio with Whisper (mlx-whisper, runs locally on Apple
   Silicon, free). Whisper captures EVERYTHING actually spoken, ads included,
   with per-word timestamps.
2. Align the clean transcript's words against Whisper's words as two token
   sequences (difflib). Ads exist only in the Whisper stream, so they match
   nothing in the transcript and are simply dropped. Transcript words that were
   really spoken land on real timestamps; the rest are interpolated.
3. Chunk the transcript into readable sentence-level cues and write .srt.

Usage
-----
    python align_transcript.py AUDIO TRANSCRIPT [-o OUT.srt] [options]

The Whisper transcription is cached next to the audio as
`<audio>.whisper.json` so re-runs (e.g. tweaking cue formatting) are instant.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


# --------------------------------------------------------------------------- #
# Transcription (Whisper via mlx-whisper)
# --------------------------------------------------------------------------- #
def transcribe(audio_path: str, model: str, language: str, cache_path: str,
               force: bool = False) -> dict:
    """Return Whisper output {'segments': [...], 'words': [{word,start,end}]}.

    Caches the raw result to `cache_path` so we only pay for transcription once.
    """
    if os.path.exists(cache_path) and not force:
        print(f"[transcribe] using cache {cache_path}", file=sys.stderr)
        with open(cache_path) as f:
            return json.load(f)

    import mlx_whisper  # imported lazily so --help etc. don't need it

    print(f"[transcribe] running Whisper ({model}) on {audio_path} ...",
          file=sys.stderr)
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        verbose=None,
    )

    # Flatten to a compact word list plus keep segments for reference.
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    out = {
        "language": result.get("language", language),
        "duration": result.get("segments", [{}])[-1].get("end") if result.get("segments") else None,
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in result.get("segments", [])
        ],
        "words": words,
    }
    with open(cache_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[transcribe] {len(words)} words, cached to {cache_path}",
          file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Normalization + tokenization
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[0-9\W_]+", re.UNICODE)


def norm_token(tok: str) -> str:
    """Normalize a word for matching: lowercase, strip accents & punctuation.

    Accent-stripping makes matching robust to Whisper's accentuation quirks
    while keeping ñ->n etc. Digits are dropped (Whisper is inconsistent about
    '20' vs 'veinte'); such tokens become empty and are treated as unalignable.
    """
    tok = unicodedata.normalize("NFKD", tok)
    tok = "".join(c for c in tok if not unicodedata.combining(c))
    tok = tok.lower()
    tok = re.sub(r"[^a-z0-9]", "", tok)
    return tok


def whisper_word_tokens(words: list[dict]) -> tuple[list[str], list[dict]]:
    """Split Whisper words into normalized tokens keeping (start,end) per token."""
    norms, meta = [], []
    for w in words:
        n = norm_token(w["word"])
        if not n:
            continue
        norms.append(n)
        meta.append(w)
    return norms, meta


# --------------------------------------------------------------------------- #
# Transcript parsing
# --------------------------------------------------------------------------- #
@dataclass
class Tok:
    """One transcript word (or a run of unspoken punctuation attached to it)."""
    text: str            # display text (original, with punctuation/spacing hints)
    norm: str            # normalized form ("" if unalignable)
    para: int            # paragraph index
    sent: int            # sentence index within the whole doc
    start: Optional[float] = None
    end: Optional[float] = None
    w_idx: Optional[int] = None  # index into whisper word list, if matched


# Spans that appear in the transcript but are NOT spoken words we can align:
#   [Speaker]:  ... speaker labels
#   (SOUNDBITE ...) ... stage directions
_SPEAKER_RE = re.compile(r"^\s*\[[^\]]*\]\s*:?")
_PAREN_RE = re.compile(r"\([^)]*\)")


def parse_transcript(
    path: str,
) -> tuple[list[Tok], list[str], dict[int, str], dict[int, list[int]]]:
    """Parse into a flat token list plus the list of raw paragraph strings.

    Alignable tokens get a non-empty `.norm`; speaker labels and parenthetical
    stage directions are kept for display but marked unalignable (norm="").

    Also returns `sent_text` (sentence id -> verbatim sentence text, as split
    by `split_sentences`) and `para_sents` (paragraph index -> ordered list of
    its sentence ids) so callers can fall back to sentence-granularity cuts
    for paragraphs that have no internal blank-line breaks to split on.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    toks: list[Tok] = []
    sent_counter = 0
    sent_text: dict[int, str] = {}
    para_sents: dict[int, list[int]] = {}

    for pi, para in enumerate(paragraphs):
        # Pull off a leading speaker label so its bracketed name isn't aligned.
        label = ""
        m = _SPEAKER_RE.match(para)
        body = para
        if m:
            label = m.group(0).strip()
            body = para[m.end():].strip()

        # Attach the speaker label to the paragraph's first token as display
        # prefix, but it contributes no alignable content.
        label_prefix = (label + " ") if label else ""

        # Split body into sentences for cue granularity.
        sentences = split_sentences(body)
        first_word_in_para = True
        for sent in sentences:
            sent_counter += 1
            sent_text[sent_counter] = sent
            para_sents.setdefault(pi, []).append(sent_counter)
            # Walk the sentence, separating parenthetical stage directions.
            pos = 0
            for pm in _PAREN_RE.finditer(sent):
                _emit_words(sent[pos:pm.start()], pi, sent_counter, toks,
                            label_prefix if first_word_in_para else "")
                if toks:
                    first_word_in_para = False
                # stage direction: display-only, unalignable
                toks.append(Tok(text=pm.group(0), norm="", para=pi, sent=sent_counter))
                pos = pm.end()
            n_before = len(toks)
            _emit_words(sent[pos:], pi, sent_counter, toks,
                        label_prefix if first_word_in_para else "")
            if len(toks) > n_before:
                first_word_in_para = False

    return toks, paragraphs, sent_text, para_sents


def _emit_words(text: str, pi: int, si: int, toks: list[Tok], prefix: str) -> None:
    """Tokenize a run of plain text into word Toks, applying an optional prefix
    (the speaker label) to the very first emitted word."""
    parts = text.split()
    for w in parts:
        n = norm_token(w)
        disp = w
        if prefix:
            disp = prefix + w
            prefix = ""  # only once
        toks.append(Tok(text=disp, norm=n, para=pi, sent=si))


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[¿¡A-ZÁÉÍÓÚÑ0-9\"“])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def align(toks: list[Tok], w_norms: list[str], w_meta: list[dict]) -> dict:
    """Assign start/end to alignable transcript tokens via sequence matching,
    then interpolate the rest. Returns match statistics."""
    # Indices of alignable transcript tokens and their normalized forms.
    g_idx = [i for i, t in enumerate(toks) if t.norm]
    g_norms = [toks[i].norm for i in g_idx]

    sm = SequenceMatcher(a=g_norms, b=w_norms, autojunk=False)
    matched = 0
    for a0, b0, size in sm.get_matching_blocks():
        for k in range(size):
            t = toks[g_idx[a0 + k]]
            wm = w_meta[b0 + k]
            t.start = wm["start"]
            t.end = wm["end"]
            t.w_idx = b0 + k
            matched += 1

    _interpolate(toks)
    return {
        "alignable": len(g_idx),
        "matched": matched,
        "match_rate": (matched / len(g_idx)) if g_idx else 0.0,
    }


def _interpolate(toks: list[Tok]) -> None:
    """Fill start/end for tokens with no match, by linear interpolation between
    the nearest anchored neighbors (in reading order). Leading/trailing gaps
    clamp to the nearest anchor."""
    n = len(toks)
    anchors = [i for i, t in enumerate(toks) if t.start is not None]
    if not anchors:
        # Nothing matched at all — degenerate; leave times unset.
        return

    # Clamp before first / after last anchor.
    first, last = anchors[0], anchors[-1]
    for i in range(first):
        toks[i].start = toks[i].end = toks[first].start
    for i in range(last + 1, n):
        toks[i].start = toks[i].end = toks[last].end

    # Interpolate interior gaps.
    for a, b in zip(anchors, anchors[1:]):
        if b == a + 1:
            continue
        t0 = toks[a].end
        t1 = toks[b].start
        gap = b - a
        for j in range(1, gap):
            frac = j / gap
            t = t0 + (t1 - t0) * frac
            toks[a + j].start = t
            toks[a + j].end = t


# --------------------------------------------------------------------------- #
# Cue building + SRT output
# --------------------------------------------------------------------------- #
@dataclass
class Cue:
    start: float
    end: float
    text: str


def build_cues(toks: list[Tok], max_chars: int, max_dur: float,
               min_dur: float, hard_max_dur: float) -> list[Cue]:
    """Group tokens into subtitle cues, breaking on sentence boundaries and
    when a cue would get too long (by characters or duration)."""
    cues: list[Cue] = []
    cur: list[Tok] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        text = _join_tokens(cur)
        if not text.strip():
            cur = []
            return
        timed = [t for t in cur if t.start is not None]
        if timed:
            start = min(t.start for t in timed)
            end = max(t.end for t in timed)
        else:
            start = cues[-1].end if cues else 0.0
            end = start
        cues.append(Cue(start=start, end=end, text=text))
        cur = []

    def cur_span() -> float:
        s = next((x.start for x in cur if x.start is not None), None)
        e = next((x.end for x in reversed(cur) if x.end is not None), None)
        return (e - s) if (s is not None and e is not None) else 0.0

    prev_sent = None
    for t in toks:
        if cur and t.sent != prev_sent:
            # Sentence boundary: flush if the cue is already big enough.
            if len(_join_tokens(cur)) >= max_chars or cur_span() >= max_dur:
                flush()
        cur.append(t)
        prev_sent = t.sent

        # Hard caps mid-sentence: by characters, and by TIME. The time cap keeps
        # stretched regions (e.g. ad-break promos whose spoken form diverges from
        # the transcript) from collapsing into one enormous cue.
        if len(_join_tokens(cur)) >= max_chars * 1.6 or cur_span() >= max_dur * 1.6:
            flush()
    flush()

    # Clamp cues whose text is short but whose span is huge — these are the
    # ad-break promos where one transcript line maps to a long spoken segment.
    # We'd rather show the line briefly and leave the promo audio un-subtitled
    # than freeze one line on screen for a minute+.
    for c in cues:
        if c.end - c.start > hard_max_dur:
            c.end = c.start + hard_max_dur

    _fix_overlaps(cues, min_dur)
    return cues


def _join_tokens(toks: list[Tok]) -> str:
    """Reconstruct readable text from display tokens with sensible spacing."""
    out = ""
    for t in toks:
        s = t.text
        if not out:
            out = s
        elif re.match(r"^[,.;:!?…)»\]]", s):
            out += s
        else:
            out += " " + s
    return out.strip()


def _fix_overlaps(cues: list[Cue], min_dur: float) -> None:
    """Ensure monotonic, non-overlapping, minimally-long cues."""
    for i, c in enumerate(cues):
        if c.end < c.start:
            c.end = c.start
        if c.end - c.start < min_dur:
            c.end = c.start + min_dur
        if i + 1 < len(cues) and c.end > cues[i + 1].start:
            # trim to next start but keep a hair of separation
            c.end = max(c.start, cues[i + 1].start - 0.01)


def fmt_ts(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_ts(c.start)} --> {fmt_ts(c.end)}\n{c.text}\n\n")


# --------------------------------------------------------------------------- #
# Ad detection + audio trimming (for LingQ's own timestamp generator)
# --------------------------------------------------------------------------- #
def detect_ad_regions(toks: list[Tok], w_meta: list[dict], duration: float,
                      min_dur: float, min_density: float,
                      merge_gap: float) -> list[tuple[float, float]]:
    """Find [start, end] time spans that are ADS: stretches of audio Whisper
    transcribed as dense speech but which the transcript never matched.

    Music, archival soundbites, and protest chants are ALSO unmatched, but they
    are sparse (few words per second), so a word-density threshold separates
    real ad reads (~3 w/s) from story audio (<1 w/s). Sparse gaps are kept —
    cutting them would drop story audio the transcript covers.
    """
    matched_w = sorted(t.w_idx for t in toks if t.w_idx is not None)
    if not matched_w:
        return []

    regions: list[tuple[float, float]] = []
    N = len(w_meta)
    matched_set = set(matched_w)

    def flush_run(i: int, j: int):
        """Consider the unmatched whisper-word run [i, j) as a possible ad."""
        t0 = w_meta[i]["start"]
        t1 = w_meta[j - 1]["end"]
        span = t1 - t0
        density = (j - i) / span if span > 0 else 0
        if span >= min_dur and density >= min_density:
            regions.append((t0, t1))

    i = 0
    while i < N:
        if i in matched_set:
            i += 1
            continue
        j = i
        while j < N and j not in matched_set:
            j += 1
        flush_run(i, j)
        i = j

    # Edge regions: audio before the first / after the last matched word is ad
    # or music. Always drop it (nothing in the transcript maps there).
    first_t = w_meta[matched_w[0]]["start"]
    last_t = w_meta[matched_w[-1]]["end"]
    if first_t > min_dur:
        regions.append((0.0, first_t))
    if duration - last_t > min_dur:
        regions.append((last_t, duration))

    # Merge regions separated by tiny gaps (spurious single-word matches inside
    # an ad block) into one contiguous cut.
    regions.sort()
    merged: list[tuple[float, float]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged


def kept_segments(regions: list[tuple[float, float]],
                  duration: float) -> list[tuple[float, float]]:
    """Complement of the ad regions over [0, duration] — the audio we keep."""
    keep = []
    cursor = 0.0
    for a, b in regions:
        a = max(0.0, a)
        if a > cursor:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        keep.append((cursor, duration))
    return [(a, b) for a, b in keep if b - a > 0.05]


def remap_time(t: float, regions: list[tuple[float, float]]) -> Optional[float]:
    """Map an original timestamp onto the trimmed timeline. Returns None if the
    time falls inside a removed (ad) region."""
    removed = 0.0
    for a, b in regions:
        if t >= b:
            removed += (b - a)
        elif t >= a:
            return None  # inside a cut
        else:
            break
    return t - removed


def trim_audio(audio_in: str, audio_out: str,
               keep: list[tuple[float, float]]) -> None:
    """Concatenate the kept audio segments into a new file via ffmpeg."""
    import subprocess

    parts, labels = [], []
    for idx, (a, b) in enumerate(keep):
        parts.append(
            f"[0:a]atrim=start={a:.3f}:end={b:.3f},"
            f"asetpts=PTS-STARTPTS[a{idx}]")
        labels.append(f"[a{idx}]")
    filt = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(keep)}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-y", "-i", audio_in, "-filter_complex", filt,
           "-map", "[out]", "-c:a", "libmp3lame", "-q:a", "4", audio_out]
    print(f"[trim] ffmpeg concat of {len(keep)} segments -> {audio_out}",
          file=sys.stderr)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clean_transcript(paragraphs: list[str], toks: list[Tok],
                     regions: list[tuple[float, float]]) -> list[str]:
    """Drop paragraphs whose audio was cut out (ad/promo-only text), so the
    remaining transcript matches the trimmed audio. Kept paragraphs are returned
    verbatim (original formatting preserved)."""
    # median aligned time per paragraph (alignable tokens only)
    from statistics import median
    times: dict[int, list[float]] = {}
    for t in toks:
        if t.norm and t.start is not None:
            times.setdefault(t.para, []).append(t.start)
    kept = []
    for pi, para in enumerate(paragraphs):
        ts = times.get(pi)
        if not ts:
            # No alignable words (pure stage direction) — keep with neighbors.
            kept.append(para)
            continue
        if remap_time(median(ts), regions) is not None:
            kept.append(para)
    return kept


# --------------------------------------------------------------------------- #
# Chunking into LingQ-sized lessons (LingQ auto-splits anything over 6000 words,
# and attaches the FULL audio to every part — so we split text AND audio here).
# --------------------------------------------------------------------------- #
def build_chunks(paragraphs: list[str], toks: list[Tok],
                 regions: list[tuple[float, float]], clean_dur: float,
                 word_limit: int, sent_text: dict[int, str],
                 para_sents: dict[int, list[int]]) -> list[dict]:
    """Split kept paragraphs into LingQ lessons. Primary break points are the
    interior ad breaks (natural episode pauses); any section still over
    word_limit is then balance-split by word count, falling back to
    sentence-boundary cuts for paragraphs that have no blank-line breaks of
    their own to split on. Each chunk carries its audio [a, b] span on the
    trimmed timeline."""
    import bisect
    import math
    from statistics import median

    torig: dict[int, list[float]] = {}      # original times per paragraph
    pmin: dict[int, list[float]] = {}        # remapped (trimmed) start times
    for t in toks:
        if t.norm and t.start is not None:
            torig.setdefault(t.para, []).append(t.start)
            r = remap_time(t.start, regions)
            if r is not None:
                pmin.setdefault(t.para, []).append(r)

    kept: list[tuple[int, str, Optional[float]]] = []
    for pi, para in enumerate(paragraphs):
        ts = torig.get(pi)
        if not ts:
            kept.append((pi, para, None))   # stage direction — travels along
            continue
        if remap_time(median(ts), regions) is not None:
            st = min(pmin[pi]) if pmin.get(pi) else None
            kept.append((pi, para, st))

    # Interior ad breaks -> split points on the trimmed timeline. An ad's start
    # maps to the seam where it was removed; keep only breaks with content on
    # both sides (drops pre-roll @0 and post-roll @end).
    split_pts = []
    for a, _ in regions:
        removed_before = sum(bb - aa for aa, bb in regions if bb <= a)
        cp = a - removed_before
        if 1.0 < cp < clean_dur - 1.0:
            split_pts.append(cp)
    split_pts.sort()

    # Assign paragraphs to ad-delimited sections (stage directions inherit the
    # current section).
    sections: list[list[tuple[int, str, Optional[float]]]] = \
        [[] for _ in range(len(split_pts) + 1)]
    cur_sec = 0
    for pi, para, st in kept:
        if st is not None:
            cur_sec = bisect.bisect_right(split_pts, st)
        sections[cur_sec].append((pi, para, st))

    def expand_oversized(
        pi: int, para: str, st: Optional[float]
    ) -> list[tuple[int, str, Optional[float]]]:
        """A paragraph with no internal blank-line breaks has no cut point of
        its own. If it alone exceeds word_limit, fall back to splitting it at
        sentence boundaries (each piece keeps a real timestamp) so the
        balance-split below has somewhere to cut."""
        if len(para.split()) <= word_limit:
            return [(pi, para, st)]
        sids = para_sents.get(pi, [])
        if len(sids) <= 1:
            return [(pi, para, st)]
        label = ""
        m = _SPEAKER_RE.match(para)
        if m:
            label = m.group(0).strip()
        sent_start: dict[int, float] = {}
        for t in toks:
            if t.para == pi and t.norm and t.start is not None:
                r = remap_time(t.start, regions)
                if r is not None:
                    sent_start.setdefault(t.sent, r)
        units = []
        for i, sid in enumerate(sids):
            text = sent_text.get(sid, "").strip()
            if not text:
                continue
            if i == 0 and label:
                text = f"{label} {text}"
            units.append((pi, text, sent_start.get(sid)))
        return units if units else [(pi, para, st)]

    # Sub-split any section that still exceeds the word limit (balanced).
    chunks: list[list[tuple[int, str, Optional[float]]]] = []
    for sec in sections:
        if not sec:
            continue
        expanded: list[tuple[int, str, Optional[float]]] = []
        for pi, para, st in sec:
            expanded.extend(expand_oversized(pi, para, st))
        sec = expanded

        words = sum(len(p.split()) for _, p, _ in sec)
        if words <= word_limit:
            chunks.append(sec)
            continue
        n = math.ceil(words / word_limit)
        target = words / n
        cur: list[tuple[int, str, Optional[float]]] = []
        acc = 0
        cuts = 0
        for pi, para, st in sec:
            cur.append((pi, para, st))
            acc += len(para.split())
            if cuts < n - 1 and acc >= target * (cuts + 1):
                chunks.append(cur)
                cur = []
                cuts += 1
        if cur:
            chunks.append(cur)

    def cstart(ch) -> Optional[float]:
        for _, _, st in ch:
            if st is not None:
                return st
        return None

    def join_text(ch: list[tuple[int, str, Optional[float]]]) -> str:
        # Blank line between distinct source paragraphs (matches the
        # original verbatim join); plain space between sentence-level
        # pieces of the same paragraph produced by expand_oversized above.
        pieces = []
        prev_pi = None
        for pi, text, _ in ch:
            if not pieces:
                pieces.append(text)
            elif pi != prev_pi:
                pieces.append("\n\n" + text)
            else:
                pieces.append(" " + text)
            prev_pi = pi
        return "".join(pieces)

    out = []
    for i, ch in enumerate(chunks):
        a = 0.0 if i == 0 else max(0.0, (cstart(chunks[i]) or 0.0) - 0.25)
        if i + 1 < len(chunks):
            b = max(a, (cstart(chunks[i + 1]) or clean_dur) - 0.25)
        else:
            b = clean_dur
        out.append({
            "text": join_text(ch),
            "a": a, "b": b,
            "words": sum(len(p.split()) for _, p, _ in ch),
        })
    return out


def cut_audio_segment(audio_in: str, audio_out: str, a: float, b: float) -> None:
    """Extract [a, b] seconds from an audio file (accurate output-side seek)."""
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", audio_in, "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
           "-c:a", "libmp3lame", "-q:a", "4", audio_out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("transcript")
    ap.add_argument("-o", "--out", help="output .srt (default: <transcript>.srt)")
    ap.add_argument("-m", "--model",
                    default="mlx-community/whisper-large-v3-turbo",
                    help="mlx-whisper model repo (default large-v3-turbo)")
    ap.add_argument("-l", "--language", default="es")
    ap.add_argument("--max-chars", type=int, default=90,
                    help="soft max characters per cue")
    ap.add_argument("--max-dur", type=float, default=7.0,
                    help="soft max seconds per cue")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="min seconds per cue")
    ap.add_argument("--hard-max-dur", type=float, default=12.0,
                    help="absolute max seconds a cue may stay on screen; longer "
                         "spans (untranscribed promos/ads) are clamped, leaving "
                         "a gap")
    ap.add_argument("--force-transcribe", action="store_true")
    ap.add_argument("--cache", help="path for Whisper cache json")
    # Ad-stripping mode: produce clean audio+text for LingQ's own timestamp
    # generator (LingQ ignores imported SRT timestamps).
    ap.add_argument("--trim-audio", action="store_true",
                    help="also write an ad-free <audio>.clean.mp3 + matching "
                         "<transcript>.clean.txt so LingQ's Generate Timestamps "
                         "works (recommended for LingQ)")
    ap.add_argument("--ad-min-dur", type=float, default=15.0,
                    help="min seconds for an unmatched span to count as an ad")
    ap.add_argument("--ad-min-density", type=float, default=1.2,
                    help="min Whisper words/sec for an unmatched span to count "
                         "as spoken ad (below this = music/chants, kept)")
    ap.add_argument("--ad-merge-gap", type=float, default=8.0,
                    help="merge ad spans separated by less than this many secs")
    ap.add_argument("--split-words", type=int, default=5800,
                    help="with --trim-audio, split into balanced LingQ lessons "
                         "each <= this many words (LingQ's own limit is 6000). "
                         "0 disables splitting")
    args = ap.parse_args(argv)

    out = args.out or (os.path.splitext(args.transcript)[0] + ".srt")
    cache = args.cache or (args.audio + ".whisper.json")

    wh = transcribe(args.audio, args.model, args.language, cache,
                    force=args.force_transcribe)
    w_norms, w_meta = whisper_word_tokens(wh["words"])
    duration = wh.get("duration") or (w_meta[-1]["end"] if w_meta else 0.0)

    toks, paragraphs, sent_text, para_sents = parse_transcript(args.transcript)
    stats = align(toks, w_norms, w_meta)
    print(f"[align] transcript alignable words: {stats['alignable']}, "
          f"matched: {stats['matched']} ({stats['match_rate']*100:.1f}%)",
          file=sys.stderr)
    print(f"[align] whisper words: {len(w_meta)}", file=sys.stderr)

    cues = build_cues(toks, args.max_chars, args.max_dur, args.min_dur,
                      args.hard_max_dur)
    write_srt(cues, out)
    print(f"[write] {len(cues)} cues -> {out}", file=sys.stderr)

    if args.trim_audio:
        regions = detect_ad_regions(toks, w_meta, duration, args.ad_min_dur,
                                    args.ad_min_density, args.ad_merge_gap)
        keep = kept_segments(regions, duration)
        removed = sum(b - a for a, b in regions)
        print(f"[trim] detected {len(regions)} ad region(s), "
              f"{removed:.0f}s removed, {duration - removed:.0f}s kept:",
              file=sys.stderr)
        for a, b in regions:
            print(f"        cut {fmt_ts(a)} - {fmt_ts(b)} ({b - a:.0f}s)",
                  file=sys.stderr)

        clean_audio = os.path.splitext(args.audio)[0] + ".clean.mp3"
        clean_txt = os.path.splitext(args.transcript)[0] + ".clean.txt"
        clean_srt = os.path.splitext(args.transcript)[0] + ".clean.srt"

        trim_audio(args.audio, clean_audio, keep)

        kept_paras = clean_transcript(paragraphs, toks, regions)
        with open(clean_txt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(kept_paras) + "\n")
        print(f"[trim] {len(kept_paras)}/{len(paragraphs)} paragraphs kept "
              f"-> {clean_txt}", file=sys.stderr)

        # Remap token times onto the trimmed timeline, drop cut tokens, rebuild
        # an SRT that matches clean.mp3 (bonus / for verification).
        ctoks = []
        for t in toks:
            if t.start is None:
                continue
            ns = remap_time(t.start, regions)
            ne = remap_time(t.end, regions)
            if ns is None or ne is None:
                continue
            t2 = Tok(text=t.text, norm=t.norm, para=t.para, sent=t.sent,
                     start=ns, end=max(ns, ne))
            ctoks.append(t2)
        ccues = build_cues(ctoks, args.max_chars, args.max_dur, args.min_dur,
                           args.hard_max_dur)
        write_srt(ccues, clean_srt)
        print(f"[trim] {len(ccues)} cues on trimmed timeline -> {clean_srt}",
              file=sys.stderr)

        # Build the LingQ-ready lesson(s) (text + matching audio) and always
        # place them in lingq_import/ — whether the episode needed splitting
        # or not — so there's a single place to look for what to import.
        if args.split_words > 0:
            trimmed_dur = kept_segments(regions, duration)
            trimmed_dur = sum(b - a for a, b in trimmed_dur)
            chunks = build_chunks(paragraphs, toks, regions, trimmed_dur,
                                  args.split_words, sent_text, para_sents)
            stem = os.path.splitext(os.path.basename(args.transcript))[0]
            outdir = os.path.join(os.path.dirname(args.transcript) or ".",
                                  "lingq_import")
            os.makedirs(outdir, exist_ok=True)
            n = len(chunks)
            if n > 1:
                print(f"[split] {sum(c['words'] for c in chunks)} words -> "
                      f"{n} lessons in {outdir}/", file=sys.stderr)
            else:
                print(f"[split] {chunks[0]['words']} words <= "
                      f"{args.split_words}; one lesson -> {outdir}/",
                      file=sys.stderr)
            for i, ch in enumerate(chunks, 1):
                name = f"{stem} (Part {i} of {n})" if n > 1 else stem
                base = os.path.join(outdir, name)
                with open(base + ".txt", "w", encoding="utf-8") as f:
                    f.write(ch["text"] + "\n")
                cut_audio_segment(clean_audio, base + ".mp3", ch["a"], ch["b"])
                # per-part SRT on the part's local timeline (verification aid)
                part_toks = []
                for t in ctoks:
                    if ch["a"] <= (t.start or -1) < ch["b"] + 0.5:
                        part_toks.append(Tok(
                            text=t.text, norm=t.norm, para=t.para,
                            sent=t.sent, start=t.start - ch["a"],
                            end=max(0.0, t.end - ch["a"])))
                pcues = build_cues(part_toks, args.max_chars, args.max_dur,
                                   args.min_dur, args.hard_max_dur)
                write_srt(pcues, base + ".srt")
                if n > 1:
                    print(f"        Part {i}/{n}: {ch['words']} words, "
                          f"{fmt_ts(ch['a'])}-{fmt_ts(ch['b'])} "
                          f"({ch['b'] - ch['a']:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
