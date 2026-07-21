# Aligned transcripts for LingQ (ad-aware)

Sync a clean source transcript to its audio for LingQ — even when the audio has
ads/promos the transcript doesn't (or vice-versa).

Everything runs **locally and free** on Apple Silicon via
[`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
No API keys, no per-minute cost. First run downloads a Whisper model (~1.5 GB)
from Hugging Face once.

## ⚠️ Important: how LingQ actually handles timestamps

LingQ **ignores timestamps in an imported `.srt`.** Its only reliable audio-text
sync is its own *Generate Timestamps* button, and that button only works when the
**audio matches the lesson text**. Ads break that match — which is the whole
problem here.

So the recommended path is **not** "import an SRT". It's: **strip the ads out of
the audio** (this tool does that automatically, using the alignment) so the audio
matches the transcript, then let LingQ generate its own timestamps. Use
`--trim-audio` and follow the LingQ steps below. The plain `.srt` is still
produced for use in other players (VLC, mpv, YouTube, etc.).

## Recommended workflow for LingQ

```bash
source .venv/bin/activate
python align_transcript.py "Episode.mp3" "Episode.txt" --trim-audio
```

This writes:

- `Episode.clean.mp3` — the audio with ad regions removed
- `Episode.clean.txt` — the transcript with ad/promo-only lines removed (so it
  matches `clean.mp3`)
- `Episode.clean.srt` — subtitles aligned to `clean.mp3` (bonus, for other players)

- `lingq_import/` — the ready-to-import lesson(s), always here regardless of
  length:
  - Under the word limit: `lingq_import/Episode.txt` / `.mp3` / `.srt`
  - Over the word limit: `lingq_import/Episode (Part N of M).txt` / `.mp3` /
    `.srt`, split into LingQ-sized parts (see below)

Then in LingQ, **for each lesson in `lingq_import/`**:

1. **Import** → paste the `.txt` (or upload it) as the lesson text.
2. Attach the matching `.mp3` as the lesson audio.
3. Wait for the audio to finish uploading, then click **Generate Timestamps**.
4. Open **Sentence Mode** — audio and text line up.

Because the ads are gone and each lesson's audio matches its text, LingQ's
matcher has an easy, exact job.

### Why it's pre-split (and split where it is)

LingQ auto-splits any imported lesson over **6,000 words** into 6,000-word parts —
**but it attaches the full audio to every part**, so part 2's text ends up paired
with audio starting at 0:00 and *Generate Timestamps* misaligns. To avoid that,
this tool splits the text **and** the audio itself into matching parts, each under
the limit, and always stages the result in `lingq_import/` — even a
single-lesson episode gets a copy there, so that folder is always the one place
to look for what's ready to import.

Split points are the **interior ad breaks** — the natural episode pauses ("Una
pausa y volvemos"). If a resulting section is still over the word limit, it's
balance-split by word count at paragraph boundaries, falling back to sentence
boundaries for transcripts with no blank-line paragraph breaks of their own.
Tune with `--split-words` (default 5800; `0` disables all of this, including
staging in `lingq_import/`).

## How it works

1. **Transcribe** the audio with Whisper → word-level timestamps for *everything
   actually spoken*, ads included. Cached to `<audio>.whisper.json`.
2. **Align** the clean transcript's words against Whisper's words as two token
   sequences (`difflib`). Ads live only in the Whisper stream, so they match
   nothing in the transcript and are dropped. Spoken transcript words land on
   real timestamps; gaps are interpolated between anchors.
3. **Chunk** the transcript into readable, sentence-level cues and write `.srt`.

This is why the intro ads and the outro music are simply absent from the output,
and why the first cue starts on the first real word of the episode.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-whisper          # requires ffmpeg on PATH (brew/macports)
```

## Usage

```bash
source .venv/bin/activate
python align_transcript.py AUDIO.mp3 TRANSCRIPT.txt
# -> writes TRANSCRIPT.srt
```

Common options:

| flag | default | meaning |
|------|---------|---------|
| `-o OUT.srt` | `<transcript>.srt` | output path |
| `-l es` | `es` | audio language (ISO code) |
| `-m REPO` | `mlx-community/whisper-large-v3-turbo` | Whisper model. Use `whisper-large-v3-mlx` for max accuracy (slower), `whisper-medium-mlx` for speed. |
| `--max-chars 90` | 90 | soft max characters per cue |
| `--max-dur 7` | 7 | soft max seconds per cue |
| `--hard-max-dur 12` | 12 | a cue never stays on screen longer than this; longer spans (untranscribed promos) are clamped, leaving a gap |
| `--trim-audio` | off | also emit ad-free `clean.mp3` + matching `clean.txt` + `clean.srt` (the LingQ path) |
| `--ad-min-dur 15` | 15 | min seconds for an unmatched span to count as an ad |
| `--ad-min-density 1.2` | 1.2 | min Whisper words/sec to call a span a spoken ad; below this = music/chants, kept |
| `--split-words 5800` | 5800 | split into LingQ lessons each ≤ this many words (LingQ's limit is 6000); `0` disables |
| `--force-transcribe` | off | ignore the cached transcription and redo it |

### How ad-stripping decides what to cut

An "ad" is a stretch of audio Whisper heard as **dense speech** that the
transcript never matched. Music, archival soundbites, and protest chants are also
unmatched but are **sparse** (few words/sec), so the density threshold keeps them
(they're story audio) and cuts only real ad reads. Pre-roll before the first
matched word and audio after the last are always dropped. Check the `[trim]` log:
it prints exactly which time ranges were cut.

Re-running is instant because the transcription is cached. Only the first run
(transcription) is slow — roughly real-time-ish on an M2 (a ~50 min episode
takes a few minutes).

## Reusing on other files

It's fully generic:

```bash
python align_transcript.py "Some Other Episode.mp3" "Some Other Episode.txt" -l es
```

The transcript format it expects is what Radio Ambulante publishes: plain UTF-8,
paragraphs separated by blank lines, optional `[Speaker]:` labels at the start of
a paragraph, and `(STAGE DIRECTIONS)` in parentheses. Speaker labels and stage
directions are kept in the displayed subtitle but excluded from word matching.

## A note on ads (what this handles well, and its one soft spot)

- **Ads only in the audio** (pre-roll, dynamically-inserted mid-roll spots,
  outro): dropped cleanly. 👍
- **Content only in the transcript** (e.g. a house promo the audio replaced with
  a different ad): it can't be timed accurately because the matching audio isn't
  there. It gets interpolated between the surrounding real anchors and clamped by
  `--hard-max-dur`, so it shows briefly in roughly the right region. This only
  affects ad-break filler, never the actual story, and the alignment snaps back
  to exact the moment real content resumes.

Check the stderr line `matched: N (XX%)` after a run — for clean episodes expect
~85–90%. A much lower number means the language code is wrong or the transcript
doesn't match the audio.
