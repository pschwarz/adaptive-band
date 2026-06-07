"""Staged backing band on top of Magenta RealTime 2 (MRT2).

A different interaction model from hello_world.py's continuous open-loop stream.
Here the app runs in stages:

  1. Drums + tempo start and never drop out.
  2. Learning stage: after a one-bar count-in, listen to the directly-wired (DI)
     live instrument for N bars and capture the CHORD PROGRESSION being played.
  3. Generate a backing track that FOLLOWS the captured progression, looping it
     forever under continuous drums until Ctrl+C.

v1 is the one-shot pipeline (drums -> learn -> generate -> loop). The north star
is continuous re-learning (a living jam partner) and, later, gesture cues
(a camera "nod" -> new progression for verse/chorus) -- both out of scope here.

This module is a NEW entry point and does NOT modify hello_world.py; it imports
the reusable pieces (beat grid, input capture, pitch front-end, key estimator,
device resolvers, the MRT2 loader) and adds chord detection + a progression-driven
playback loop.

MRT2 has no symbolic chord input. The only pitch lever is generate(notes=<128 ints>),
a per-MIDI-pitch state map (-1 masked / 0 off / 1 on / 2 onset / 3 on, model's
choice). We follow a chord by driving that mask per chord: chord tones get a strong
bias, the rest of the (stable) key a mild bias, out-of-key is masked. See chord_mask.
"""

import argparse
import contextlib
import select
import sys
import termios
import threading
import time
import tty

from hello_world import (
    BEAT_GRID,
    MODE_ALIASES,
    MODE_INTERVALS,
    NOTE_NAMES,
    InputCapture,
    MagentaRT2SystemMlxfn,
    basic_pitch_histogram,
    beat_plan,
    estimate_key,
    mode_name,
    prompt_with_tempo,
    resolve_input_device,
    resolve_output_device,
    tempo_word,
)

FPS = 25.0  # MRT2 generates at 25 frames/sec (40ms frames)

# Defaults for the BACKING bake (make_backing_loop) = Magenta's reference params (see
# magenta_rt/mlx/generate.py). The earlier "let the model breathe" values (faint prompt +
# wide top-k) sounded worse than the library examples; A/B confirmed the reference params.
# Applied only to the backing (NOT the drum bake); explicit
# --style-strength/--cfg-notes/--temperature/--top-k overrides per-knob.
BACKING_CFG_MUSICCOCA = 3.0  # firm prompt guidance (Magenta default)
BACKING_CFG_NOTES = 1.0      # push the lone root-note hint (Magenta default)
BACKING_TEMPERATURE = 1.3    # Magenta default
BACKING_TOP_K = 40           # Magenta default
BACKING_MAX_TAKE_SECONDS = 6.0  # longest single backing take per chord; longer chords
                                # loop this take (crossfaded) instead of generating more
MRT2_FPS = 25.0                 # MRT2 generates 25 frames/sec (40ms frames); see beat_plan
BACKING_ONSET_GRANULARITY = "beat"  # where the backing places note ONSETS so the model locks
                                    # to tempo: "beat" = onset every beat (strong grid),
                                    # "bar" = onset only on beat 1 of each bar (subtler)
MIX_HEADROOM = 0.7              # fixed scale applied to BOTH layers when summing drums+backing,
                                # so their sum stays within full-scale without a per-beat
                                # limiter. A per-beat divide-by-peak would duck the drums only
                                # on beats where the backing's onset clips -- an amplitude
                                # wobble synced to the backing that reads as a groove change.

# --- Chord vocabulary -----------------------------------------------------
# Triads only for v1 (maj/min/dim/aug). Richer qualities (sevenths) are the job
# of the deferred Sonnet-labeling upgrade -- see learn_progression's note. Each
# value is the set of semitone offsets from the chord root.
CHORD_QUALITIES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}

# Per-chord-tone weights for the matching templates. Root + fifth are heaviest so
# the matcher leans on the chord's skeleton (the third disambiguates quality but
# is the tone most often masked by overtones/bleed), mirroring the tonic/fifth
# emphasis of hello_world's key templates (_DEGREE_WEIGHT).
_CHORD_TONE_WEIGHT = {0: 3.0, 7: 2.0}   # root, fifth (offset from chord root)
_CHORD_DEFAULT_WEIGHT = 1.5             # third (and the #5 of an aug, etc.)


def chord_templates():
    """{(root 0..11, quality): zero-meaned 12-vector} for all 48 triads. Matched
    against a beat's mean chroma (also zero-meaned) by dot product -- the same
    correlation trick estimate_key uses for keys, but over chord shapes."""
    import numpy as np

    templates = {}
    for root in range(12):
        for quality, offsets in CHORD_QUALITIES.items():
            t = np.zeros(12)
            for off in offsets:
                t[(root + off) % 12] = _CHORD_TONE_WEIGHT.get(off, _CHORD_DEFAULT_WEIGHT)
            templates[(root, quality)] = t - t.mean()
    return templates


def match_chord(chroma12, templates):
    """Mean chroma (12-vector) -> best-fitting (root, quality). Zero-means the
    chroma and picks the template with the highest correlation."""
    import numpy as np

    c = np.asarray(chroma12, dtype=np.float64)
    c = c - c.mean()
    if not float(np.linalg.norm(c)):
        return None
    return max(templates, key=lambda k: float((c * templates[k]).sum()))


def chord_mask(chord_root, chord_quality, key_root, key_mode):
    """(chord) + (stable key) -> 128-int notes vector for generate().

    Graded soft bias (strategy B): chord tones = 3 (strong, model's choice of
    onset/continuation), other in-key scale degrees = 1 (mild on), everything
    out of key = -1 (masked / "no opinion"). No hard 0 ("off") -- a continuous
    hard lock under the streaming state decays coherence within ~10s (see the
    encoding note in hello_world.py). The chord leads; the key keeps the model
    melodically free within the scale; chromatic tones are merely un-biased."""
    key_mode = MODE_ALIASES.get(key_mode, key_mode)
    chord_pcs = {(chord_root + off) % 12 for off in CHORD_QUALITIES[chord_quality]}
    key_pcs = {(key_root + iv) % 12 for iv in MODE_INTERVALS[key_mode]}
    mask = []
    for p in range(128):
        pc = p % 12
        if pc in chord_pcs:
            mask.append(3)
        elif pc in key_pcs:
            mask.append(1)
        else:
            mask.append(-1)
    return mask


def root_hint_mask(root_pc, on=3):
    """A DELIBERATELY GENTLE notes vector for generate(): only ONE pitch is set --
    the chord root in the middle octave (MIDI 60+pc, the C4..B4 band) -- and every
    other pitch is -1 ("no opinion", model free). A broad key/tonal-center hint, not
    a harmonic cage: the old graded chord+key mask (chord_mask) sounded bad because it
    biased the whole chord+scale at once; this just anchors the root and lets the model
    breathe everywhere else. Chord QUALITY is carried by the prompt, not this mask (a
    single root can't encode maj/min). `on` is the pitch state (3 = on, model's choice
    of onset/continuation -- the right value for a standalone static hint; see the
    notes encoding in magenta_rt mlx system.generate)."""
    mask = [-1] * 128
    mask[60 + root_pc] = on
    return mask


def chord_name(root, quality):
    return f"{NOTE_NAMES[root]}{'' if quality == 'maj' else quality}"


# Spell out the chord quality in words MusicCoCa is likely to have seen in text.
_QUALITY_WORD = {"maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented"}


def chord_prompt_phrase(root, quality):
    """A natural-language chord name to splice into the style prompt, e.g.
    'C major' / 'A minor'. Used instead of the notes mask: MRT2's only real lever is
    the text style, and a graded notes mask tended to make the backing sound bad, so
    we just tell MusicCoCa the chord(s) in words."""
    return f"{NOTE_NAMES[root]} {_QUALITY_WORD[quality]}"


# Map quality spellings a user might type to our internal quality keys.
_QUALITY_PARSE = {
    "": "maj", "maj": "maj", "major": "maj", "M": "maj",
    "min": "min", "minor": "min", "m": "min", "-": "min",
    "dim": "dim", "diminished": "dim", "o": "dim",
    "aug": "aug", "augmented": "aug", "+": "aug",
}


def parse_chord(token):
    """Parse one user-typed chord like 'C', 'Dminor', 'F#m', 'Bbaug', 'E#' into
    (root, quality). Root is a note letter A-G with an optional # or b accidental
    (enharmonics fold to a pitch class: E# -> F, Cb -> B). The rest is the quality
    (see _QUALITY_PARSE; empty == major). Raises ValueError on garbage."""
    s = token.strip()
    if not s:
        raise ValueError("empty chord")
    letter = s[0].upper()
    if letter not in "ABCDEFG":
        raise ValueError(f"{token!r}: must start with a note letter A-G")
    pc = NOTE_NAMES.index({"A": "A", "B": "B", "C": "C", "D": "D",
                           "E": "E", "F": "F", "G": "G"}[letter])
    i = 1
    while i < len(s) and s[i] in "#b":
        pc = (pc + (1 if s[i] == "#" else -1)) % 12
        i += 1
    qual = s[i:]
    if qual not in _QUALITY_PARSE:
        raise ValueError(f"{token!r}: unknown chord quality {qual!r}")
    return pc, _QUALITY_PARSE[qual]


def parse_chords(spec, beats_per_bar):
    """Parse a comma-separated chord spec (e.g. 'C, C, Dminor, C, E#, C') into a
    per-bar progression: a list of {start_beat,length_beats,root,quality} segments,
    one bar (beats_per_bar beats) per chord, tiling [0, n*beats_per_bar). This is the
    --backing-only counterpart to learn_progression's output, built from text instead
    of live chord detection so the backing stage can run with no DI/learning."""
    tokens = [t for t in (tok.strip() for tok in spec.split(",")) if t]
    if not tokens:
        raise ValueError("no chords given")
    prog = []
    for bar, tok in enumerate(tokens):
        root, quality = parse_chord(tok)
        prog.append({"start_beat": bar * beats_per_bar,
                     "length_beats": beats_per_bar,
                     "root": root, "quality": quality})
    return prog


def bar_chords(progression, total_beats, beats_per_bar):
    """The chord of each bar in the phrase: (root, quality) at each bar's downbeat.
    progression is per-beat segments tiling [0, total_beats); we sample the chord on
    each bar's first beat. Returns a list of length total_beats // beats_per_bar."""
    n_bars = total_beats // beats_per_bar
    return [_chord_at_beat(progression, bar * beats_per_bar, total_beats)
            for bar in range(n_bars)]


def chord_spans(progression, total_beats, beats_per_bar):
    """Collapse the progression into chord SPANS measured in BEATS: a list of
    (root, quality, n_beats), merging adjacent equal-root beats into one span. Sum of
    n_beats == total_beats.

    This is the unit make_backing_loop generates over: one take per chord CHANGE, however
    many beats the chord is held. Sampling per beat (via _chord_at_beat) means a chord that
    changes mid-bar splits correctly. ROOT-ONLY: quality normalized to 'maj' so a
    maj<->min on the same root doesn't split a span (the backing keys off the root; quality
    rides in the prompt)."""
    spans = []
    for beat in range(total_beats):
        root, _quality = _chord_at_beat(progression, beat, total_beats)
        quality = "maj"
        if spans and spans[-1][0] == root:
            r, q, n = spans[-1]
            spans[-1] = (r, q, n + 1)
        else:
            spans.append((root, quality, 1))
    return spans


def phrase_prompt(band_prompt, bars):
    """Build a whole-phrase style prompt naming the chord of every bar in order, e.g.
    '<band prompt>, chord progression by bar: C major, A minor, F major, G major'.
    `bars` is a list of (root, quality). MRT2 has no symbolic chord input, so this is
    how we tell it the harmony for the whole N-bar phrase in one shot."""
    seq = ", ".join(chord_prompt_phrase(r, q) for r, q in bars)
    return f"{band_prompt}, chord progression by bar: {seq}"


# --- Stage 2: learning ----------------------------------------------------

def _beats_per_bar(time_sig):
    return BEAT_GRID[time_sig][0]


def logical_beats(bpm, time_sig):
    """Wrap beat_plan so each yield is one whole logical beat, not beat_plan's
    onset/tail split. beat_plan yields a backbeat as two tuples (a 1-frame drum
    onset + an (N-1)-frame tail); we coalesce them so callers count beats and
    advance chords once per beat.

    Yields (chunks, bar_start) where chunks is a list of (frames, drums) to feed
    generate() in order for this beat (one item normally, two for a split backbeat),
    and bar_start is True on beat 1 of the bar."""
    plan = beat_plan(bpm, time_sig)
    while True:
        frames, drums, bar_start = next(plan)
        chunks = [(frames, drums)]
        if drums == [1] and frames == 1:
            # Backbeat onset: the next yield is its tail (same beat, bar_start False).
            tail_frames, tail_drums, _ = next(plan)
            chunks.append((tail_frames, tail_drums))
        yield chunks, bar_start


# A long, emphatic DRUMS-ONLY style prompt for baking the loop. The band prompt
# (e.g. "indie rock ballad") is deliberately NOT used here: with notes=None alone,
# MRT2 still renders the full kit-plus-band that the band words imply, leaking
# piano/guitar/bass into what should be a bare drum bed. MRT2's only real lever is
# the text style, so we describe ONLY percussion, repeatedly and exclusively, and
# explicitly forbid pitched instruments. tempo/time-sig hint is appended for groove.
def drum_prompt(bpm, time_sig):
    n = int(bpm) if float(bpm).is_integer() else bpm
    base = ("solo drum kit, drums only, acoustic drum set, kick snare hi-hat, "
            "tight steady backbeat groove, dry studio drums, no melody, no chords, "
            "no bass, no guitar, no piano, no synth, no vocals, percussion only, "
            "isolated drum track, drum stem")
    hint = f"{base}, {tempo_word(bpm)}, {n} BPM"
    if time_sig:
        hint += f", {time_sig} time"
    return hint


def _crossfade_wrap(samples, fade_len):
    """Make a buffer loop seamlessly by crossfading its lead-out (the region just
    PAST the loop point) back over its head. `samples` is (loop + lead-out); we
    take the first `loop_len = len(samples) - fade_len` samples as the loop body,
    then equal-power crossfade the `fade_len` samples that follow the loop point
    (the model's natural continuation) into the first `fade_len` samples of the
    loop. The result is a `loop_len`-long buffer whose end flows into its start.

    Why this kills the seam: baked beat-by-beat with state carried forward, the
    buffer's true end never anticipates jumping back to sample 0, so the wrap pops.
    The lead-out IS what the model would have played next, so fading it over the
    head splices the loop's end onto its beginning with matching energy/phase."""
    import numpy as np

    loop_len = samples.shape[0] - fade_len
    body = samples[:loop_len].copy()
    leadout = samples[loop_len:loop_len + fade_len]
    head = body[:fade_len]
    # Equal-power (cos/sin) fade: out-going leadout fades down, in-coming head up.
    t = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t * (np.pi / 2.0))[:, None]
    fade_in = np.sin(t * (np.pi / 2.0))[:, None]
    body[:fade_len] = leadout * fade_out + head * fade_in
    return body


def _crossfade_join(prev_tail, new_bar, fade_len):
    """Splice two consecutive streamed bars together so the join doesn't click.
    Unlike _crossfade_wrap (which folds a buffer's lead-out back over its OWN head
    for a seamless loop), here we have two separate bars generated in sequence: the
    previous bar's last `fade_len` samples (`prev_tail`) and the new bar (`new_bar`).
    We equal-power crossfade prev_tail into the head of new_bar IN PLACE, so new_bar
    starts from the previous bar's energy/phase instead of cold. Same cos/sin curve
    as _crossfade_wrap. Returns new_bar (mutated copy of its head). fade_len must be
    <= len(prev_tail) and <= len(new_bar)."""
    import numpy as np

    out = new_bar.copy()
    t = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t * (np.pi / 2.0))[:, None]
    fade_in = np.sin(t * (np.pi / 2.0))[:, None]
    out[:fade_len] = prev_tail[-fade_len:] * fade_out + out[:fade_len] * fade_in
    return out


def _summarize_notes(notes):
    """Compact one-line summary of a 128-int notes mask (or None) for logging: the
    full vector is too noisy, so we report which pitch classes are set to what level."""
    if notes is None:
        return "None"
    if len(notes) <= 8:                     # short masks (e.g. drums [0]/[1]) print whole
        return repr(list(notes))
    levels = {}                             # level -> [pitch indices set to it]
    for i, v in enumerate(notes):
        if v != -1:                         # -1 == masked == default, skip the noise
            levels.setdefault(int(v), []).append(i)
    if not levels:
        return f"all -1 (masked), len={len(notes)}"
    parts = [f"={lvl}:{idxs}" for lvl, idxs in sorted(levels.items())]
    return f"len={len(notes)} non-default[{', '.join(parts)}]"


def _gen(mrt, tag, **kw):
    """Log every parameter sent to mrt.generate(), then call it. `tag` labels the call
    site (drums / backing span). All generation goes through here so the exact knobs
    feeding Magenta are visible each time."""
    fields = []
    for k, v in kw.items():
        if k == "style":
            fields.append("style=<embedding>")
        elif k == "notes":
            fields.append(f"notes={_summarize_notes(v)}")
        elif k == "state":
            fields.append(f"state={'None' if v is None else '<carried>'}")
        else:
            fields.append(f"{k}={v}")
    print(f"  [generate:{tag}] " + ", ".join(fields))
    return mrt.generate(**kw)


def make_drum_loop(mrt, bpm, time_sig, bars, cfg_musiccoca=None,
                   temperature=None, top_k=None, return_clip=False):
    """Generate ONE fixed drum-only loop of `bars` bars, ONCE, and return it as a
    single contiguous sample buffer plus per-logical-beat sample offsets.

    The whole point: MRT2 composes percussion fresh on every generate() call, so
    streaming it beat-by-beat makes the drums wander/drift. Instead we bake a short
    loop here and replay that exact buffer forever (DrumLoopFeeder), so the groove
    is rock-steady and identical every bar. We embed a dedicated DRUMS-ONLY prompt
    (drum_prompt), NOT the band prompt, so no pitched instruments leak in.

    The whole phrase is baked as ONE continuous _fn pass (_bake_drums_onsets) with a
    per-FRAME drums-onset map -- a drum onset on each backbeat's first frame, masked
    (model's choice) elsewhere -- not one generate() per beat. This is the same
    onset-driven, single-pass approach the backing uses (_bake_take_onsets), so the
    groove locks tempo/meter without per-beat restart artifacts.

    Seamless wrap: we bake one EXTRA bar past the loop as a "lead-out" (the model's
    natural continuation), then crossfade that lead-out back over the loop's head
    (_crossfade_wrap) so jumping from the last beat to the first has no pop.

    Returns (samples, sample_rate, channels, beat_offsets) where samples is an
    (N, channels) float32 array, and beat_offsets is a list of (start, end) sample
    indices for each logical beat in the loop (len == bars * beats_per_bar). The
    last beat's end == N, so concatenating all slices reproduces the loop exactly
    and wrapping from the last beat back to the first is seamless. With return_clip,
    also returns the raw baked take (loop + lead-out, pre-wrap) for --generation-debug."""
    import numpy as np

    bpb = _beats_per_bar(time_sig)
    total_beats = bars * bpb
    leadout_beats = bpb  # one extra bar, baked then folded into the head as a fade
    gen_beats = total_beats + leadout_beats

    _dprompt = drum_prompt(bpm, time_sig)
    print(f"  [embed:drums] prompt={_dprompt!r}")
    style_tokens = mrt.tokenize_style(mrt.embed_style(_dprompt)).tolist()

    # Backbeat onset frames + per-beat frame bounds over the whole bake (loop + lead-out).
    onset_frames, beat_bounds = _drum_onset_frame_plan(bpm, time_sig, gen_beats)
    total_frames = beat_bounds[-1][1]
    print(f"  [generate:drums] frames={total_frames}, onsets={len(onset_frames)}, "
          f"state=None, drums=[1]@backbeats/[-1]else, notes=masked, "
          f"cfg_musiccoca={cfg_musiccoca}, temperature={temperature}, top_k={top_k}")
    t0 = time.time()
    take, _state, sample_rate, channels = _bake_drums_onsets(
        mrt, style_tokens, total_frames, onset_frames, None,
        cfg_musiccoca, temperature, top_k)
    print(f"  [drums] {bars}-bar loop gen {time.time() - t0:.2f}s")

    spf = take.shape[0] / total_frames       # samples per frame (~1920 @48k)
    # True per-loop-beat sample bounds (exclude the lead-out beats). Rebuild contiguously so
    # rounding can't leave a gap: each beat ends where the next starts, last ends at loop_len.
    starts = [round(s * spf) for s, _e in beat_bounds[:total_beats]]
    loop_len = round(beat_bounds[total_beats - 1][1] * spf)
    beat_offsets = [(starts[i], starts[i + 1] if i + 1 < len(starts) else loop_len)
                    for i in range(len(starts))]
    fade_len = take.shape[0] - loop_len      # the baked lead-out length
    # Cap the crossfade so it never bleeds past the first beat (keeps beat_offsets
    # valid: only samples inside beat 0 are altered).
    fade_len = min(fade_len, beat_offsets[0][1])
    trimmed = take[:loop_len + fade_len]
    samples = _crossfade_wrap(trimmed, fade_len)
    if return_clip:
        return samples, sample_rate, channels, beat_offsets, take
    return samples, sample_rate, channels, beat_offsets


class DrumLoopFeeder:
    """Replays a baked drum loop (from make_drum_loop) one logical beat at a time,
    wrapping forever. push_next_beat(player) pushes the next beat's slice to the
    queued player and returns whether that beat is a bar start (beat 0 of the loop's
    bar grid). Carries no model state -- pure buffer playback, so the groove never
    drifts."""

    def __init__(self, samples, beat_offsets, beats_per_bar):
        self.samples = samples
        self.beat_offsets = beat_offsets
        self.bpb = beats_per_bar
        self.i = 0  # index into beat_offsets, wraps modulo len

    def next_slice(self):
        """Return (samples_for_this_beat, bar_start) and advance, wrapping forever.
        Does not touch the player -- lets a caller mix this slice with another layer
        (see LayeredFeeder) before pushing."""
        n = len(self.beat_offsets)
        start, end = self.beat_offsets[self.i % n]
        bar_start = (self.i % n) % self.bpb == 0
        self.i += 1
        return self.samples[start:end], bar_start

    def push_next_beat(self, player):
        sl, bar_start = self.next_slice()
        player.push_samples(sl)
        return bar_start


def bar_prompt(band_prompt, root, quality):
    """Single-bar style prompt naming one chord, e.g. '<band prompt>, current chord:
    C major'. make_backing_loop embeds this per chord SPAN so the prompt carries the
    span's chord QUALITY (the root-note mask can't encode maj/min)."""
    return f"{band_prompt}, current chord: {chord_prompt_phrase(root, quality)}"


def _chord_at_beat(progression, beat, total_beats):
    """Which progression segment covers `beat` (0-based, wrapping at total_beats)?
    Returns (root, quality). Segments tile [0, total_beats) contiguously."""
    b = beat % total_beats
    for seg in progression:
        if seg["start_beat"] <= b < seg["start_beat"] + seg["length_beats"]:
            return seg["root"], seg["quality"]
    return progression[-1]["root"], progression[-1]["quality"]


def _beat_frame_counts(bpm, time_sig, n_beats):
    """The exact frame count of each of the first n_beats logical beats, matching
    beat_plan's fractional-frame accumulator. We sum the chunks of each logical beat
    so a backing slice generated per-beat lines up sample-for-sample with the drum
    loop's same-index beat (both derive from the same deterministic beat_plan)."""
    beats = logical_beats(bpm, time_sig)
    counts = []
    for _ in range(n_beats):
        chunks, _ = next(beats)
        counts.append(sum(frames for frames, _drums in chunks))
    return counts


def _take_beats(bpm, time_sig, span_beats, max_seconds):
    """How many of a span's leading beats fit in max_seconds of generated audio (>=1,
    <= span_beats). Frames -> seconds at MRT2_FPS. This is the per-chord take length we
    actually generate; a chord held longer than this loops the take instead of generating
    more. Always returns >=1 so even a very fast tempo yields a real take."""
    frames = _beat_frame_counts(bpm, time_sig, span_beats)
    acc = 0
    n = 0
    for bf in frames:
        if n >= 1 and (acc + bf) / MRT2_FPS > max_seconds:
            break
        acc += bf
        n += 1
    return n


def _onset_frame_plan(bpm, time_sig, n_beats, granularity):
    """Per-FRAME onset plan for a take of n_beats. Returns (onset_frames, beat_bounds):
    onset_frames is a set of frame indices that should carry a note ONSET (the first frame
    of each onset-bearing beat); beat_bounds is [(start_frame, end_frame), ...] per beat.
    granularity "beat" -> every beat's first frame; "bar" -> only beat 1 of each bar."""
    bpb = _beats_per_bar(time_sig)
    counts = _beat_frame_counts(bpm, time_sig, n_beats)
    onset_frames = set()
    beat_bounds = []
    f = 0
    for k, n in enumerate(counts):
        if (granularity != "bar") or (k % bpb == 0):
            onset_frames.add(f)
        beat_bounds.append((f, f + n))
        f += n
    return onset_frames, beat_bounds


def _bake_take_onsets(mrt, style_tokens, root, total_frames, onset_frames, state,
                      cfg_musiccoca, cfg_notes, cfg_drums, temperature, top_k):
    """Generate `total_frames` frames of backing in ONE CONTINUOUS pass, driving the model's
    internal frame stepper (mrt._fn) directly so the NOTES conditioning can vary PER FRAME --
    a note ONSET (root pitch state 2) on each frame in `onset_frames`, sustain (state 1)
    elsewhere, everything else -1. This is the frame-aligned onset roll the model entrains to
    (so it locks tempo/meter) WITHOUT splicing per-beat generate() takes: one unbroken
    generation, state threaded frame->frame, the model never restarts. drums forced OFF.

    Reaches into MagentaRT2SystemMlxfn internals (._fn, ._initial_state, ._TOKEN_OFFSET,
    ._num_*, ._sample_rate) because the public generate() builds one static notes mask per
    call and so cannot carry a per-frame onset map. Mirrors generate()'s arg layout
    (mlx/system._build_mlxfn_args) and audio assembly. Returns (samples, state, sample_rate,
    channels)."""
    import numpy as np
    import mlx.core as mx

    off = mrt._TOKEN_OFFSET
    drums_tokens = [0] * mrt._num_drums                  # drums OFF
    masked_style = [-1] * len(style_tokens)

    def cond_for(notes):
        # Positive + the two CFG negatives (masked style / masked notes), as generate() builds.
        cond = mx.array((np.array(style_tokens + notes + drums_tokens, dtype=np.int32) + off
                         ).reshape(1, 1, -1), dtype=mx.int32)
        neg_mc = mx.array((np.array(masked_style + notes + drums_tokens, dtype=np.int32) + off
                           ).reshape(1, 1, -1), dtype=mx.int32)
        neg_n = mx.array((np.array(style_tokens + [-1] * len(notes) + drums_tokens, dtype=np.int32)
                          + off).reshape(1, 1, -1), dtype=mx.int32)
        return cond, neg_mc, neg_n

    scalars = [
        mx.array([temperature]),
        mx.array([top_k], dtype=mx.int32),
        mx.array([cfg_musiccoca]),
        mx.array([cfg_notes]),
        mx.array([cfg_drums]),
    ]
    forced = mx.zeros((1, 0, mrt._rvq_depth), dtype=mx.int32)

    onset_notes = root_hint_mask(root, on=2)
    sustain_notes = root_hint_mask(root, on=1)

    if state is None:
        state = list(mrt._initial_state)
    audio_frames = []
    for fi in range(total_frames):
        cond, neg_mc, neg_n = cond_for(onset_notes if fi in onset_frames else sustain_notes)
        outputs = mrt._fn([cond] + scalars + [neg_mc, neg_n, forced] + state)
        mx.eval(outputs)
        audio_frames.append(np.array(outputs[0]))       # (1, 2, 1920)
        state = list(outputs[1:])

    all_audio = np.concatenate(audio_frames, axis=-1)    # (1, 2, total_samples)
    samples = (all_audio[0].T.astype(np.float32) / 32768.0)  # (total_samples, 2)
    return np.ascontiguousarray(samples), state, mrt._sample_rate, samples.shape[1]


def _drum_onset_frame_plan(bpm, time_sig, n_beats):
    """Per-FRAME drum onset plan for a take of n_beats. Returns (onset_frames, beat_bounds):
    onset_frames is the set of frame indices that carry a drum ONSET -- the first frame of each
    BACKBEAT beat (BEAT_GRID's 1-based backbeat set, e.g. {2,4} in 4/4); beat_bounds is
    [(start_frame, end_frame), ...] per beat. Sibling of _onset_frame_plan but backbeat-aware,
    so it mirrors beat_plan's old drums=[1]-on-backbeats placement."""
    bpb = _beats_per_bar(time_sig)
    backbeats = BEAT_GRID[time_sig][1]               # 1-based beat numbers within a bar
    counts = _beat_frame_counts(bpm, time_sig, n_beats)
    onset_frames = set()
    beat_bounds = []
    f = 0
    for k, n in enumerate(counts):
        if (k % bpb) + 1 in backbeats:               # 1-based beat-in-bar
            onset_frames.add(f)                      # onset on the beat's first frame
        beat_bounds.append((f, f + n))
        f += n
    return onset_frames, beat_bounds


def _bake_drums_onsets(mrt, style_tokens, total_frames, onset_frames, state,
                       cfg_musiccoca, temperature, top_k):
    """Generate `total_frames` frames of DRUMS in ONE CONTINUOUS pass, driving mrt._fn directly so
    the DRUMS conditioning can vary PER FRAME -- a drum ONSET (drums [1]) on each frame in
    `onset_frames` (the backbeats), masked ([-1] = model's choice, the _fn equivalent of the old
    drums=None tails/fills) elsewhere. Notes are masked off every frame (drums-only loop, no pitched
    material). This is the drum counterpart of _bake_take_onsets (which instead varies NOTES and
    forces drums off): one unbroken generation, state threaded frame->frame, no per-beat restart.

    cfg_musiccoca/temperature/top_k arrive raw from the CLI and may be None; unlike the public
    generate() this manual path doesn't resolve them, so we fall back to the model's instance
    defaults here. cfg_notes/cfg_drums use the instance defaults too (1.0) -- matching the old
    generate()-based drum bake, NOT the backing's stronger cfg_drums. Returns (samples, state,
    sample_rate, channels)."""
    import numpy as np
    import mlx.core as mx

    cfg_musiccoca = mrt.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
    temperature = mrt.temperature if temperature is None else temperature
    top_k = mrt.top_k if top_k is None else top_k

    off = mrt._TOKEN_OFFSET
    notes = [-1] * mrt._num_notes                        # notes OFF (drums-only)
    masked_style = [-1] * len(style_tokens)

    def cond_for(drums):
        cond = mx.array((np.array(style_tokens + notes + drums, dtype=np.int32) + off
                         ).reshape(1, 1, -1), dtype=mx.int32)
        neg_mc = mx.array((np.array(masked_style + notes + drums, dtype=np.int32) + off
                           ).reshape(1, 1, -1), dtype=mx.int32)
        neg_n = mx.array((np.array(style_tokens + [-1] * len(notes) + drums, dtype=np.int32)
                          + off).reshape(1, 1, -1), dtype=mx.int32)
        return cond, neg_mc, neg_n

    scalars = [
        mx.array([temperature]),
        mx.array([top_k], dtype=mx.int32),
        mx.array([cfg_musiccoca]),
        mx.array([mrt.cfg_notes]),
        mx.array([mrt.cfg_drums]),
    ]
    forced = mx.zeros((1, 0, mrt._rvq_depth), dtype=mx.int32)

    onset_drums = [1] * mrt._num_drums                   # drum onset
    fill_drums = [-1] * mrt._num_drums                   # masked = model's choice (old drums=None)

    if state is None:
        state = list(mrt._initial_state)
    audio_frames = []
    for fi in range(total_frames):
        cond, neg_mc, neg_n = cond_for(onset_drums if fi in onset_frames else fill_drums)
        outputs = mrt._fn([cond] + scalars + [neg_mc, neg_n, forced] + state)
        mx.eval(outputs)
        audio_frames.append(np.array(outputs[0]))        # (1, 2, 1920)
        state = list(outputs[1:])

    all_audio = np.concatenate(audio_frames, axis=-1)    # (1, 2, total_samples)
    samples = (all_audio[0].T.astype(np.float32) / 32768.0)
    return np.ascontiguousarray(samples), state, mrt._sample_rate, samples.shape[1]


def make_backing_loop(mrt, band_prompt, bpm, time_sig, progression, key,
                      cfg_musiccoca, cfg_notes, cfg_drums, temperature, top_k,
                      crossfade=False, return_clips=False):
    """Bake the pitched backing band ONCE, ONE generate() call PER CHORD CHANGE (capped
    at ~6s of audio), carrying model state forward through the changes, then crossfade-wrap
    the whole thing into a seamless loop. Baked once -> a fixed buffer can't drift and
    playback needs no model calls.

    What shapes the sound:
    - **per-chord take capped at BACKING_MAX_TAKE_SECONDS, ONE continuous pass with a
      per-frame onset map.** chord_spans collapses the progression into spans of consecutive
      identical chords in BEATS (not bars). For each span we make ONE take of min(span, ~6s):
      a span that fits in <=6s is generated at its exact length, a longer span generates a ~6s
      take and LOOPS it (crossfaded) to fill the span. The take is generated as ONE unbroken
      stream (_bake_take_onsets drives the model's frame stepper mrt._fn directly), varying
      only the NOTES conditioning per frame to place a note ONSET at each beat's start. This
      frame-aligned onset grid is what makes the model lock to tempo/meter -- WITHOUT splicing
      per-beat takes (which restart the model and sound worse). The public generate() can't do
      this (it reuses one notes mask per call), so we step _fn ourselves.
    - **state carried frame->frame and span->span.** Within a take the model never restarts
      (continuous _fn stepping); across spans the final state seeds the next (start None), so
      the model voice-leads across the changes instead of disconnected takes butt-spliced. A
      final LEAD-OUT span repeats span 0's chord (state carried) so the wrap splices
      like-on-like.
    - **seams.** _crossfade_join smooths each span->span boundary (and the lead-out join);
      _crossfade_wrap folds a lead-out back over the head both for a long span's internal
      take-repeat AND for the whole-loop wrap -- same seam strategy as make_drum_loop.
    - **gentle root-note mask + onsets.** notes is a SINGLE pitch (middle-octave root): state
      2 (onset) on each onset frame, state 1 (sustain) elsewhere, everything else -1 ("no
      opinion"). A broad tonal-center hint with a rhythmic pulse, not the old graded chord+key
      mask (that sounded bad). cfg_notes scales it; BACKING_ONSET_GRANULARITY controls whether
      onsets land every beat or once per bar.
    - **chord QUALITY in the PROMPT.** bar_prompt names the span's chord ("...current
      chord: C major") -- the mask's lone root can't encode maj/min, so the prompt does.
    - **drums=0 (off), not None.** None == "model's choice", which let the backing compose
      its own kit; we force drums OFF (pushed by cfg_drums) so this layer is pitched-only.

    progression: list of {start_beat,length_beats,root,quality} tiling [0, total_beats).
    key is accepted for signature stability but unused (harmony = prompt + root mask).

    Returns (samples, sample_rate, channels, beat_offsets) -- same contract as
    make_drum_loop: beat_offsets has one (start,end) per logical beat over the LOOP region
    (total_beats beats), tiling contiguously, last end == len. The per-beat offsets are the
    TRUE per-beat sample bounds (from each beat's actually-generated chunks) so the unchanged
    beat-lockstep player/mixer keeps working and the grid lines up tightly; the backing wraps
    at its OWN length (not padded to the drum loop), so chord changes may slowly rotate against
    the drum grid over many loops."""
    import numpy as np

    bpb = _beats_per_bar(time_sig)
    total_beats = sum(s["length_beats"] for s in progression)

    # Spans of consecutive identical chords (in BEATS) + one lead-out span repeating span
    # 0's chord so the seamless wrap splices like-on-like.
    spans = chord_spans(progression, total_beats, bpb)
    leadout = (spans[0][0], spans[0][1], 1)  # one beat of span 0's chord
    spans_with_leadout = spans + [leadout]

    sample_rate = channels = None
    state = None
    prev_tail = None
    pieces = []          # per-span (samples, local_beat_bounds) after joins
    clips = []           # per-span RAW generated take (pre tile/splice), for --generation-debug
    for si, (root, quality, n_beats) in enumerate(spans_with_leadout):
        _bprompt = bar_prompt(band_prompt, root, quality)
        print(f"  [embed:backing span {si}] prompt={_bprompt!r}")
        style_tokens = mrt.tokenize_style(mrt.embed_style(_bprompt)).tolist()
        take_beats = _take_beats(bpm, time_sig, n_beats, BACKING_MAX_TAKE_SECONDS)

        t0 = time.time()
        # Bake the take in ONE continuous _fn pass with a per-frame onset map (no per-beat
        # splicing). For a long-held chord we bake take_beats (+1 lead-out beat to fold over
        # its head) and tile; otherwise we bake the whole span exactly.
        gen_beats = (take_beats + 1) if take_beats < n_beats else n_beats
        onset_frames, beat_bounds = _onset_frame_plan(bpm, time_sig, gen_beats,
                                                      BACKING_ONSET_GRANULARITY)
        total_frames = beat_bounds[-1][1]
        print(f"  [generate:backing span {si} {chord_name(root, quality)}] "
              f"frames={total_frames}, onsets={len(onset_frames)}, state="
              f"{'None' if state is None else '<carried>'}, drums=[0], notes=root@{root}, "
              f"cfg_musiccoca={cfg_musiccoca}, cfg_notes={cfg_notes}, cfg_drums={cfg_drums}, "
              f"temperature={temperature}, top_k={top_k}")
        take, state, sr, ch = _bake_take_onsets(
            mrt, style_tokens, root, total_frames, onset_frames, state,
            cfg_musiccoca, cfg_notes, cfg_drums, temperature, top_k)
        if sample_rate is None:
            sample_rate, channels = sr, ch
        spf = take.shape[0] / total_frames               # samples per frame (~1920 @48k)
        # The raw take exactly as the model generated it (before any tile/splice/wrap). The
        # lead-out span (last) is a wrap artifact, not a real chord clip, so skip it.
        if si < len(spans):
            clips.append((chord_name(root, quality), n_beats, take))

        if take_beats < n_beats:
            # Long-held chord: crossfade-wrap the take(+lead-out) into a seamless unit, then
            # tile/trim to the full span length. Per-beat bounds repeat the take's pattern.
            take_samples = round(beat_bounds[take_beats - 1][1] * spf)  # end of last take beat
            span_samples = round(sum(_beat_frame_counts(bpm, time_sig, n_beats)) * spf)
            fl = min(take.shape[0] - take_samples, take_samples) if crossfade else 0
            unit = _crossfade_wrap(take[:take_samples + fl], fl) if fl > 0 else take[:take_samples]
            unit_bounds = [(round(s * spf), round(e * spf)) for s, e in beat_bounds[:take_beats]]
            reps = -(-span_samples // unit.shape[0]) + 1  # ceil + 1 spare for the wrap fade
            tiled = np.tile(unit, (reps, 1))
            wfl = min(unit.shape[0], span_samples) if crossfade else 0  # wrap the tiled repeat too
            span = _crossfade_wrap(tiled[:span_samples + wfl], wfl) if wfl > 0 else tiled[:span_samples]
            local_bounds = []
            for k in range(n_beats):
                base = (k // take_beats) * unit.shape[0]
                s, e = unit_bounds[k % take_beats]
                local_bounds.append((min(base + s, span.shape[0]), min(base + e, span.shape[0])))
        else:
            # Span fits in <=6s: the take IS the span.
            span = take
            local_bounds = [(round(s * spf), round(e * spf)) for s, e in beat_bounds]
        print(f"  [backing] span {si} {chord_name(root, quality)} {n_beats}beat "
              f"(take {take_beats}beat) gen {time.time() - t0:.2f}s")

        # First beat of this span (samples) -- used to cap join fades so beat 0 stays valid.
        beat0 = max(1, local_bounds[0][1] - local_bounds[0][0])
        # Crossfade-join this span's head onto the previous span's tail so the seam doesn't
        # click. _crossfade_join overlaps in place (length preserved), so local_bounds stay
        # valid. Cap the fade to this span's first beat and to what the tail can supply.
        if crossfade and prev_tail is not None:
            fade_len = min(len(prev_tail), span.shape[0], beat0)
            if fade_len > 0:
                span = _crossfade_join(prev_tail, span, fade_len)
        prev_tail = span[-beat0:]
        pieces.append((span, local_bounds))

    # Concatenate the loop spans (everything but the lead-out) into the loop body, shifting
    # each span's TRUE per-beat bounds by the running cursor.
    loop_spans = pieces[:-1]
    leadout_samples = pieces[-1][0]
    body = np.concatenate([p[0] for p in loop_spans], axis=0)
    starts = []
    cursor = 0
    for span, local_bounds in loop_spans:
        for s, _e in local_bounds:
            starts.append(cursor + s)
        cursor += span.shape[0]
    # Contiguous grid: each beat runs from its true start to the next beat's start; the last
    # beat ends at the body length. (Crossfade joins / tile clamping can otherwise leave tiny
    # gaps; the player needs a gapless tiling whose last end == len.)
    beat_offsets = [(starts[i], starts[i + 1] if i + 1 < len(starts) else body.shape[0])
                    for i in range(len(starts))]
    assert len(beat_offsets) == total_beats, (len(beat_offsets), total_beats)

    if not crossfade:
        # No crossfade: hard butt-splice the loop (drop the lead-out). The end->start seam is
        # an audible cut -- that's the point, to compare against the smoothed version.
        samples = body
    else:
        # Seamless loop: append the lead-out (span 0's chord, state-continued) past the loop
        # point and crossfade it back over the head. Cap the fade to the first beat so slice 0
        # stays valid (mirrors make_drum_loop's wrap).
        fade_len = min(leadout_samples.shape[0], beat_offsets[0][1] - beat_offsets[0][0])
        trimmed = np.concatenate([body, leadout_samples[:fade_len]], axis=0)
        samples = _crossfade_wrap(trimmed, fade_len)

    if return_clips:
        return samples, sample_rate, channels, beat_offsets, clips
    return samples, sample_rate, channels, beat_offsets


def _mix_beats(a, b, backing_gain):
    """Sum two per-beat slices (drums `a` + backing `b`) into one. They may differ in
    length (drums loop and backing loop wrap at different beat counts, and beat_plan's
    fractional-frame accumulator makes a given beat index a few samples longer/shorter),
    so we mix over the shorter length and tail the longer one through unchanged.

    Both layers are scaled by a fixed MIX_HEADROOM so their sum stays within full-scale
    WITHOUT a per-beat limiter. We deliberately do NOT divide each beat by its peak: that
    would attenuate the drums only on beats where the backing's onset clips, ducking the
    kit in sync with the backing -- an amplitude wobble that's heard as the groove
    changing / a beat dropping when a loud layer comes in. A constant scale keeps every
    drum beat at the same level, so the groove stays steady."""
    import numpy as np

    n = min(a.shape[0], b.shape[0])
    out = MIX_HEADROOM * (a[:n] + backing_gain * b[:n])
    tail = MIX_HEADROOM * (a[n:] if a.shape[0] > n else backing_gain * b[n:])
    out = np.concatenate([out, tail], axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


@contextlib.contextmanager
def raw_keys():
    """Put stdin in cbreak (raw-ish) mode so single keypresses arrive immediately
    without Enter, restoring the terminal on exit. Yields a poll() -> char-or-None
    reader that never blocks. No-op (yields a reader that returns None) if stdin
    isn't a tty (e.g. piped), so the loop still runs."""
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll():
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Playback debug modes, toggled live by hotkey.
MODE_DRUMS = "drums"
MODE_BACKING = "backing"
MODE_MIXED = "mixed"
_MODE_KEYS = {"d": MODE_DRUMS, "b": MODE_BACKING, "m": MODE_MIXED}


class LayeredFeeder:
    """Replays the drum loop and the backing loop in lockstep, one beat at a time.
    Each layer wraps at its own length (drums = loop_bars; backing = full progression),
    so the same fixed drums sit under the evolving chord cycle.

    What gets pushed depends on `mode` (hotkey-toggled): MIXED sums both layers,
    DRUMS pushes only the drum slice, BACKING only the pitched slice -- the three
    debug modes. Also tracks the global playback beat so the caller can print the
    chord of the bar that's starting."""

    def __init__(self, drums, backing, progression, total_beats, beats_per_bar,
                 backing_gain=0.9, mode=MODE_MIXED):
        self.drums = drums
        self.backing = backing
        self.progression = progression
        self.total_beats = total_beats
        self.bpb = beats_per_bar
        self.backing_gain = backing_gain
        self.mode = mode
        self.beat = 0  # global playback beat, for chord lookup

    def current_chord(self):
        """(root, quality) of the chord covering the beat about to play."""
        return _chord_at_beat(self.progression, self.beat, self.total_beats)

    def push_next_beat(self, player):
        d_slice, bar_start = self.drums.next_slice()
        b_slice, _ = self.backing.next_slice()
        if self.mode == MODE_DRUMS:
            out = d_slice
        elif self.mode == MODE_BACKING:
            out = self.backing_gain * b_slice
        else:
            out = _mix_beats(d_slice, b_slice, self.backing_gain)
        player.push_samples(out)
        self.beat += 1
        return bar_start


class BackingSoloFeeder:
    """Plays ONLY the pitched backing loop, no drums -- for hearing the harmony naked
    while tuning the backing bake (used by --backing-only). Same play_band_forever
    interface as LayeredFeeder (beat / bpb / mode / current_chord / push_next_beat), but
    wraps a single beat-slice feeder (a DrumLoopFeeder over the baked backing buffer).
    `mode` is accepted and ignored so the d/b/m hotkeys don't crash; everything is the
    backing layer here."""

    def __init__(self, backing, progression, total_beats, beats_per_bar,
                 backing_gain=0.9, mode=MODE_BACKING):
        self.backing = backing
        self.progression = progression
        self.total_beats = total_beats
        self.bpb = beats_per_bar
        self.backing_gain = backing_gain
        self.mode = mode
        self.beat = 0

    def current_chord(self):
        return _chord_at_beat(self.progression, self.beat, self.total_beats)

    def push_next_beat(self, player):
        b_slice, bar_start = self.backing.next_slice()
        player.push_samples(self.backing_gain * b_slice)
        self.beat += 1
        return bar_start


def _beat_chroma(capture, bpm, min_seconds=1.6):
    """Snapshot the input ring buffer and return a 12-vector pitch-class histogram
    for the most recent beat. We hand basic-pitch a trailing slice of at least
    min_seconds (its input window is ~2s; one beat at most tempos is shorter, so a
    too-short slice gets left-padded with silence and the chord barely registers).
    A ~1.6s trailing slice keeps the current chord's tones present while staying
    dominated by the most recent playing. Reads capture._buf directly so we don't
    modify hello_world.py."""
    import numpy as np

    # Ring buffer: the write head is _pos, so the chronological order is
    # buf[_pos:] then buf[:_pos]. Roll so the most recent sample lands at the end,
    # then take the trailing slice.
    pos = capture._pos
    snap = np.concatenate([capture._buf[pos:], capture._buf[:pos]])
    want = max(int(min_seconds * capture.sr), int((60.0 / bpm) * capture.sr))
    snap = snap[-want:] if snap.shape[0] > want else snap
    if float(np.sqrt(np.mean(snap ** 2))) < capture.rms_gate:
        return np.zeros(12)  # silent beat
    return basic_pitch_histogram(snap, capture.sr, capture._note_threshold)


def _bar_root(chroma12):
    """Strongest-correlating root pitch-class for a whole bar's summed chroma,
    quality ignored. Returns int 0..11, or None if the bar was silent."""
    import numpy as np

    c = np.asarray(chroma12, dtype=np.float64)
    c = c - c.mean()
    if not float(np.linalg.norm(c)):
        return None
    templates = chord_templates()
    root, _q = max(templates, key=lambda k: float((c * templates[k]).sum()))
    return root


def _detect_repeat(roots):
    """Return loop length L (>=2) if `roots` is exactly two back-to-back copies of
    its first L bars AND that L-bar block contains >=1 root change; else None.
    Requires an even count: L = len(roots)//2, first half must equal second half."""
    n = len(roots)
    if n < 4 or n % 2:                 # need >=2 bars per cycle, 2 cycles => >=4 bars
        return None
    L = n // 2
    if roots[:L] != roots[L:]:
        return None
    if len(set(roots[:L])) < 2:        # enforce "one root change" inside the loop
        return None
    return L


# --- Queued player (self-contained; modeled on hello_world.stream_beats) ---

class QueuedPlayer:
    """Decouples MLX generation (main thread) from PortAudio playback (its own
    callback thread) via a bounded queue, with a lead cushion sized in audio
    seconds. Generation must stay on the thread that imported the .mlxfn model, so
    callers push() generated wavs from the main thread and the audio callback only
    memcpys -- the same thread discipline as hello_world.stream_beats."""

    def __init__(self, lead_seconds=1.5, output_device=None, external=False):
        import queue

        import numpy as np

        self._np = np
        self._queue = queue.Queue(maxsize=64)
        self._lead_seconds = lead_seconds
        self._output_device = output_device
        self._external = external  # True: a shared duplex stream drives render(); don't open our own
        self._stream = None
        self._carry = None
        self.sample_rate = None
        self.channels = None
        self._lead_samples = None

    def configure(self, sample_rate, channels):
        """Set sample rate/channels up front (used when samples are pushed as raw
        arrays via push_samples rather than wav objects)."""
        np = self._np
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.channels = channels
            self._carry = np.empty((0, self.channels), dtype=np.float32)
            self._lead_samples = int(self._lead_seconds * self.sample_rate)

    def push(self, wav):
        """Queue a generated wav (main thread). Learns sample rate/channels from
        the first push and lazily opens the output stream once a lead cushion of
        audio is buffered."""
        self.configure(wav.sample_rate, wav.num_channels)
        self.push_samples(wav.samples)

    def push_samples(self, samples):
        """Queue a raw (frames, channels) sample array (main thread). configure()
        must have run (or a prior push) so sample rate/channels are known. Lazily
        opens the output stream once a lead cushion of audio is buffered."""
        np = self._np
        self._queue.put(np.ascontiguousarray(samples, dtype=np.float32))
        # In external (duplex) mode the shared stream is already running and pulls
        # via render(); we never open our own OutputStream.
        if not self._external and self._stream is None \
                and self._queued_samples() >= self._lead_samples:
            self._open()

    def _queued_samples(self):
        return sum(c.shape[0] for c in list(self._queue.queue))

    def render(self, outdata, frames):
        """Fill `outdata` (frames, channels) from the queue/carry. Called from the
        audio thread -- our own OutputStream callback, or a shared duplex callback in
        external mode. Copy-only, no MLX (same thread discipline as the input path)."""
        import queue

        np = self._np
        while self._carry.shape[0] < frames:
            try:
                self._carry = np.concatenate([self._carry, self._queue.get_nowait()], axis=0)
            except queue.Empty:
                break
        n = min(frames, self._carry.shape[0])
        outdata[:n] = self._carry[:n]
        if n < frames:
            outdata[n:] = 0.0
            print("\n[underrun] generation fell behind")
        self._carry = self._carry[n:]

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"\n[audio status] {status}")
        self.render(outdata, frames)

    def _open(self):
        import sounddevice as sd

        out_idx = resolve_output_device(self._output_device) if self._output_device else None
        self._stream = sd.OutputStream(
            device=out_idx, samplerate=self.sample_rate, channels=self.channels,
            dtype="float32", blocksize=0, callback=self._callback,
        )
        self._stream.start()

    def needs_audio(self):
        """True when buffered audio is below the lead cushion (caller should
        generate more). Before the stream opens we always want more (pre-roll)."""
        if self._lead_samples is None:
            return True
        return self._queued_samples() < self._lead_samples

    def started(self):
        return self._stream is not None

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


class DuplexStream:
    """One shared full-duplex PortAudio stream on a single device (e.g. the iD4),
    so input capture and output playback use the SAME device and clock.

    The earlier two-stream design forbade this: a separate InputStream + OutputStream
    on one CoreAudio device collide with -10863. But a SINGLE duplex sd.Stream is one
    PortAudio client on one clock -- no collision, and nothing to drift against (there
    is only one clock). The callback feeds the capture's ring buffer from indata and
    fills outdata from the player's queue -- both copy-only, no MLX.

    The interface defaults to 44.1k but MRT2 audio is 48k, so we open the stream at
    the player's sample rate (48k) and align the capture's analysis rate to match
    (the iD4 supports 48k even though it reports 44.1k as default)."""

    def __init__(self, device_idx, capture, player, samplerate, in_channels):
        import sounddevice as sd

        self._sd = sd
        self._capture = capture
        self._player = player
        self._in_channels = in_channels
        # The player produces `player.channels` out channels at `samplerate`.
        self._stream = sd.Stream(
            device=device_idx, samplerate=samplerate,
            channels=(in_channels, player.channels), dtype="float32",
            blocksize=0, callback=self._callback,
        )

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"\n[audio status] {status}")
        self._capture.feed(indata, frames)   # input ring buffer (copy-only)
        self._player.render(outdata, frames)  # output from queue (copy-only)

    def start(self):
        self._stream.start()

    def close(self):
        self._stream.stop()
        self._stream.close()


class DrumKeepAlive:
    """Keeps the baked drum loop sounding on a background thread while the MAIN
    thread is busy baking the backing (make_backing_loop), AND tracks where the live
    player is in the progression so the band can enter in phase. Without this the
    output goes silent between Stage 2 (learning) and Stage 3 (play) -- the bake blocks
    the main thread for seconds.

    We assume the live player keeps cycling the progression along with the continuous
    drums. The drum loop beat index (`loop.i`) is the live-phase clock: `lock_beat` is
    where the loop locked (a bar boundary), so `loop.i - lock_beat` is the number of
    drum beats since the live player restarted at bar 1. At each bar start we print the
    ASSUMED live position (`bar X/N (assumed): <chord>`) as a continuous positional cue
    through the otherwise-blind bake gap.

    With `enter_at_top=True` the thread stops itself the instant the live phase reaches
    the top of the progression (`(loop.i - lock_beat) % total_beats == 0`) on a bar
    start -- the caller then enters the band on that exact downbeat, in phase with the
    live player. Without it, it runs until stop()/__exit__ (used while the bake runs).

    Safe to run off-thread: it only memcpys buffer slices into the player's
    thread-safe queue (loop.push_next_beat -> player.push_samples); no MLX touched
    (MLX generation stays on the main thread, which is doing the bake). stop() signals
    and joins so NO push races the Stage-3 feeder -- the main thread fully owns the
    queue again before playback starts. The drum loop index is left wherever it
    landed; it wraps, and drums<->chord phase rotating is already accepted."""

    def __init__(self, loop, player, progression=None, total_beats=None,
                 lock_beat=0, beats_per_bar=None, enter_at_top=False):
        self._loop = loop
        self._player = player
        self._progression = progression
        self._total_beats = total_beats
        self._lock_beat = lock_beat
        self._bpb = beats_per_bar
        self._enter_at_top = enter_at_top
        self._stop = threading.Event()
        self._reached_top = threading.Event()  # set when enter_at_top fires
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _live_beat(self):
        """Drum beats since the live player restarted the progression at bar 1."""
        return self._loop.i - self._lock_beat

    def _print_cue(self):
        """Print the ASSUMED live progression position at a bar start."""
        if self._progression is None or not self._total_beats:
            return
        live = self._live_beat()
        n_bars = self._total_beats // self._bpb
        bar = (live % self._total_beats) // self._bpb + 1
        root, quality = _chord_at_beat(self._progression, live, self._total_beats)
        print(f"  bar {bar}/{n_bars} (assumed): {chord_name(root, quality)}")

    def _run(self):
        while not self._stop.is_set():
            if not self._player.needs_audio():
                self._stop.wait(0.005)
                continue
            # Entering-at-top: stop ON the top-of-progression downbeat, BEFORE pushing
            # it, so loop.i sits exactly on that beat and the caller's feeder takes over
            # there with no gap or double-push.
            if self._enter_at_top and self._live_beat() % self._total_beats == 0 \
                    and self._live_beat() > 0:
                self._print_cue()
                self._reached_top.set()
                return
            bar_start = self._loop.push_next_beat(self._player)
            if bar_start:
                self._print_cue()

    def wait_for_top(self):
        """Block until the enter_at_top thread reaches the top of the progression."""
        self._reached_top.wait()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


class ProgressionLearner:
    """The clock + ears for PROGRESSIVE learning, on ONE background thread. Owns the
    drum clock (pushing the fixed loop beat-by-beat), the listening (per-beat input
    snapshot + per-bar root detection), and -- once the main thread hands it a baked
    vamp -- mixing that vamp under the drums in real time.

    Why a thread: MLX generation MUST stay on the main thread (the thread that imported
    the model). To bake the single-chord vamp on the main thread the instant bar 1's
    root is heard, the real-time clock+ears can't also be on the main thread. So this
    class is the off-thread clock (copy-only pushes into the player queue, same
    discipline as DrumKeepAlive) AND the off-thread ears (read-only numpy snapshot of
    capture._buf, the same benign race the audio thread already tolerates). It NEVER
    touches MLX.

    Lifecycle (driven by the main-thread orchestrator in main()):
      start()                 -> count-in, then listen
      wait_first_chord()      -> blocks until bar 1's root is heard (Stage B trigger)
      install_vamp(feeder)    -> main thread hands over the baked vamp; it layers in at
                                 the next bar boundary
      wait_locked()           -> blocks until the loop repeats; returns
                                 (progression, key, lock_beat)
      ... main thread bakes the full backing while this keeps drums+vamp alive,
          printing the assumed live position each bar ...
      enter_at_top()          -> ask the thread to stop ON the next top-of-progression
                                 downbeat (vamp still sounding up to it)
      wait_for_top() / stop() -> block for that downbeat, then join

    The drum loop index `loop.i` is written by ONLY this thread while it runs; the main
    thread reads it after stop()/join() (single-writer, same invariant as before)."""

    def __init__(self, loop, capture, bpm, time_sig, player,
                 count_in_bars=4, beats_per_bar=None, backing_gain=0.9):
        self._loop = loop
        self._capture = capture
        self._bpm = bpm
        self._time_sig = time_sig
        self._player = player
        self._count_in_bars = count_in_bars
        self._bpb = beats_per_bar if beats_per_bar is not None else _beats_per_bar(time_sig)
        self._backing_gain = backing_gain

        # learner -> main
        self._first_chord = threading.Event()
        self._first_root = None
        self._first_quality = "maj"   # pre-lock we only know the root; quality rides in the prompt
        self._locked = threading.Event()
        self._result = None           # (progression, key, lock_beat)

        # main -> learner
        self._vamp_lock = threading.Lock()
        self._pending_vamp = None     # installed vamp feeder, promoted on a bar boundary
        self._vamp = None             # active vamp feeder (mixed under the drums)
        self._stop = threading.Event()
        self._enter_at_top = False
        self._reached_top = threading.Event()

        # Earliest drum-beat index at which the vamp may layer in. The vamp must NOT come
        # in before: the full warm-up (count_in_bars of drums only) + 1 bar to actually
        # hear the chord being played + 1 bar to bake/align it. So the floor is 2 full
        # bars after the warm-up ends -> the vamp enters at bar (count_in_bars + 2) start
        # at the earliest (e.g. bar 6 for a 4-bar warm-up). The bake can push it later;
        # this only guarantees it never comes in EARLIER.
        self._vamp_earliest_beat = (count_in_bars + 2) * self._bpb

        self._lock_beat = 0           # set at lock; live-phase reference for cue printing
        self._thread = threading.Thread(target=self._run, daemon=True)

    # --- public API (main thread) ---

    def start(self):
        self._thread.start()

    def wait_first_chord(self):
        self._first_chord.wait()
        return self._first_root, self._first_quality

    def install_vamp(self, feeder):
        with self._vamp_lock:
            self._pending_vamp = feeder

    def wait_locked(self):
        self._locked.wait()
        return self._result

    def enter_at_top(self):
        self._enter_at_top = True

    def wait_for_top(self):
        self._reached_top.wait()

    def stop(self):
        self._stop.set()
        self._thread.join()

    # --- internals (background thread) ---

    def _push_beat(self):
        """Push ONE beat at playback pace: drums alone, or drums+vamp mixed once a vamp
        is installed. Promote a pending vamp only on a bar boundary so it enters cleanly
        on a downbeat. Returns whether this beat was a drum-loop bar start."""
        while not self._player.needs_audio() and not self._stop.is_set():
            self._stop.wait(0.005)
        with self._vamp_lock:
            if (self._pending_vamp is not None
                    and self._loop.i % self._bpb == 0
                    and self._loop.i >= self._vamp_earliest_beat):
                self._vamp = self._pending_vamp
                self._pending_vamp = None
            vamp = self._vamp
        if vamp is None:
            return self._loop.push_next_beat(self._player)
        d_slice, bar_start = self._loop.next_slice()
        b_slice, _ = vamp.next_slice()
        self._player.push_samples(_mix_beats(d_slice, b_slice, self._backing_gain))
        return bar_start

    def _print_cue(self):
        """Print the ASSUMED live progression position at a bar start (post-lock only)."""
        progression, _key, _lock = self._result
        total_beats = sum(s["length_beats"] for s in progression)
        live = self._loop.i - self._lock_beat
        n_bars = total_beats // self._bpb
        bar = (live % total_beats) // self._bpb + 1
        root, quality = _chord_at_beat(progression, live, total_beats)
        print(f"  bar {bar}/{n_bars} (assumed): {chord_name(root, quality)}")

    def _run(self):
        import numpy as np

        bpb = self._bpb
        capture = self._capture
        bpm = self._bpm
        captured_beats = []   # (pcs, chroma12) per captured beat -- feeds estimate_key

        # Phase 1 -- count-in: count_in_bars full bars of drums, no capture.
        print(f"count-in ({self._count_in_bars} bars)...")
        intro_total = self._count_in_bars * bpb
        for played in range(intro_total):
            if self._stop.is_set():
                return
            if played % bpb == 0:
                print(f"intro bar {played // bpb + 1}/{self._count_in_bars}")
            self._push_beat()

        # Phase 2 -- listen open-endedly, one root per bar, until a loop repeats.
        print("LISTENING for the loop (play it through twice)...")
        bar_chroma = np.zeros(12)
        roots = []
        loop_len = None
        while not self._stop.is_set():
            self._push_beat()
            hist = _beat_chroma(capture, bpm)
            pcs = {p for p in range(12) if hist[p] > 0}
            captured_beats.append((pcs, np.asarray(hist, dtype=np.float64)))
            bar_chroma = bar_chroma + hist
            if len(captured_beats) % bpb:
                continue  # mid-bar

            r = _bar_root(bar_chroma)
            bar_chroma = np.zeros(12)
            if r is None and roots:
                r = roots[-1]
            if r is None:
                print("bar -: silent (waiting for first chord)")
                continue
            roots.append(r)
            bar_idx = len(roots)

            # Stage B trigger: the FIRST real root heard. The main thread is blocked in
            # wait_first_chord(); it bakes the one-chord vamp and installs it.
            if not self._first_chord.is_set():
                self._first_root = r
                self._first_quality = "maj"
                self._first_chord.set()

            seq = " ".join(NOTE_NAMES[x] for x in roots)
            print(f"bar {bar_idx}: heard {NOTE_NAMES[r]}   [{seq}]")

            loop_len = _detect_repeat(roots)
            if loop_len is not None:
                roots = roots[:loop_len]
                break
        if self._stop.is_set():
            return

        # Drum beat index at lock (a bar boundary). The NEXT drum downbeat is where the
        # live player restarts the progression at bar 1 -- the live-phase reference.
        self._lock_beat = self._loop.i
        lock_beat = self._lock_beat
        print(f"[locked] {loop_len}-bar loop: " + " ".join(NOTE_NAMES[x] for x in roots))

        # Build progression + key from the locked roots (same as the old learn_progression).
        total_chroma = np.sum([c for _, c in captured_beats], axis=0)
        key = estimate_key(total_chroma, modal_margin=capture._modal_margin)
        kr, km = key
        quality = "min" if MODE_ALIASES.get(km, km) == "aeolian" else "maj"
        filled = [(r, quality) for r in roots for _ in range(bpb)]
        progression = []
        for i, ch in enumerate(filled):
            if progression and progression[-1]["root"] == ch[0] \
                    and progression[-1]["quality"] == ch[1]:
                progression[-1]["length_beats"] += 1
            else:
                progression.append({"start_beat": i, "length_beats": 1,
                                    "root": ch[0], "quality": ch[1]})
        print(f"\n[key] {NOTE_NAMES[kr]} {mode_name(km)}")
        print("[progression] " + " | ".join(
            f"{chord_name(s['root'], s['quality'])}x{s['length_beats']}" for s in progression))

        self._result = (progression, key, lock_beat)
        total_beats = sum(s["length_beats"] for s in progression)
        self._locked.set()

        # Phase 3 -- post-lock tail: keep drums+vamp going (the master clock) while the
        # main thread bakes the full backing. Print the assumed live position each bar.
        # When asked to enter-at-top, stop ON the next top-of-progression downbeat so the
        # caller's band takes over there, in phase, with the vamp sounding right up to it.
        while not self._stop.is_set():
            if not self._player.needs_audio():
                self._stop.wait(0.005)
                continue
            live = self._loop.i - lock_beat
            if self._enter_at_top and live % total_beats == 0 and live > 0:
                self._print_cue()
                self._reached_top.set()
                return
            bar_start = self._push_beat()
            if bar_start:
                self._print_cue()


def play_band_forever(feeder, label, player):
    """Replay the band (LayeredFeeder) forever. Hotkeys (no Enter): d=drums-only,
    b=backing-only, m=mixed, q/Ctrl+C=quit. Prints the chord of each bar as it
    starts. No model calls -- pure buffer playback (+ per-beat sum)."""
    print(f"{label}")
    print("  keys: [d] drums  [b] backing  [m] mixed  [q] quit "
          f"(now: {feeder.mode})")
    try:
        with raw_keys() as poll:
            while True:
                key = poll()
                if key in ("q", "\x03"):  # q or Ctrl+C
                    break
                if key in _MODE_KEYS:
                    feeder.mode = _MODE_KEYS[key]
                    print(f"  -> mode: {feeder.mode}")
                if not player.needs_audio():
                    time.sleep(0.01)
                    continue
                # Print the chord of the bar that's about to start, before pushing it.
                # Bar shown as position WITHIN the progression (bar X/N) + loop count, so a
                # repeat of the phrase is obvious (bar resets to 1/N, loop ticks up).
                if feeder.beat % feeder.bpb == 0:
                    root, quality = feeder.current_chord()
                    n_bars = feeder.total_beats // feeder.bpb
                    bar = (feeder.beat % feeder.total_beats) // feeder.bpb + 1
                    loop = feeder.beat // feeder.total_beats + 1
                    print(f"  bar {bar}/{n_bars} (loop {loop}): {chord_name(root, quality)}")
                feeder.push_next_beat(player)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")
        player.close()


def play_band_once(feeder, label, player):
    """Play the baked progression EXACTLY ONCE (total_beats beats) and stop -- the
    --backing-only --no-repeat path. No forever-loop, so you hear only what was baked
    without the loop-point crossfade-wrap repeating. Pushes all beats (the player drains
    them from its queue), waits for the queue to empty, then stops. q/Ctrl+C aborts early.
    No model calls -- pure buffer playback."""
    print(f"{label}")
    print(f"  playing once ({feeder.total_beats} beats), no loop. [q] quit")
    try:
        with raw_keys() as poll:
            # Push every beat of the single progression pass.
            while feeder.beat < feeder.total_beats:
                if poll() in ("q", "\x03"):
                    return
                if not player.needs_audio():
                    time.sleep(0.01)
                    continue
                if feeder.beat % feeder.bpb == 0:
                    root, quality = feeder.current_chord()
                    n_bars = feeder.total_beats // feeder.bpb
                    bar = feeder.beat // feeder.bpb + 1
                    print(f"  bar {bar}/{n_bars}: {chord_name(root, quality)}")
                feeder.push_next_beat(player)
            # Wait for the queued audio to actually play out before stopping.
            while player._queued_samples() > 0:
                if poll() in ("q", "\x03"):
                    return
                time.sleep(0.02)
            time.sleep(player._lead_seconds)  # let the lead-cushion carry drain
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")
        player.close()


def play_clips_with_silence(clips, sample_rate, channels, label, player, gap_seconds=1.0):
    """--generation-debug: play each RAW generated clip ONCE, in order, with `gap_seconds`
    of silence between them, then stop. No tiling, splicing, or looping -- you hear exactly
    what the model produced per chord span. q/Ctrl+C aborts. Pure buffer playback."""
    import numpy as np

    player.configure(sample_rate, channels)
    gap = np.zeros((int(gap_seconds * sample_rate), channels), dtype=np.float32)
    lead = int(player._lead_seconds * sample_rate)  # queued floor == when this clip starts sounding
    print(f"{label}")
    print(f"  {len(clips)} clip(s), {gap_seconds:.0f}s silence between. [q] quit")
    try:
        with raw_keys() as poll:
            for i, (name, n_beats, take) in enumerate(clips):
                if i:
                    player.push_samples(gap)
                player.push_samples(take)
                # Wait until everything BEFORE this clip has played out (queue drained down to
                # the lead buffer) -- i.e. this clip is what's now starting to sound. Log THEN,
                # so the printed chord matches the audio you hear (the "we're in F now" cue).
                while player._queued_samples() > take.shape[0] + lead:
                    if poll() in ("q", "\x03"):
                        return
                    time.sleep(0.02)
                print(f"  clip {i + 1}/{len(clips)}: {name} "
                      f"({n_beats} beat, {take.shape[0] / sample_rate:.1f}s)")
            while player._queued_samples() > 0:
                if poll() in ("q", "\x03"):
                    return
                time.sleep(0.02)
            time.sleep(player._lead_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")
        player.close()


def main():
    p = argparse.ArgumentParser(description="Staged backing band (drums -> learn -> loop).")
    p.add_argument("--prompt", default="warm ambient synth pads")
    p.add_argument("--tempo", type=float, default=100.0,
                   help="target tempo in BPM (drives beat grid + soft prompt hint)")
    p.add_argument("--time-sig", choices=tuple(BEAT_GRID), default="4/4",
                   help="time signature (drives backbeat grid + soft prompt hint)")
    p.add_argument("--size", default="mrt2_base")
    p.add_argument("--learn-bars", type=int, default=16,
                   help="DEPRECATED/ignored: learning is now adaptive (listens "
                        "until the played loop repeats)")
    p.add_argument("--count-in-bars", type=int, default=4,
                   help="bars of drums-only intro before learning starts")
    p.add_argument("--vamp", action="store_true",
                   help="progressive vamp: layer a single-chord backing under the drums "
                        "as soon as the first chord is heard, before the full progression "
                        "locks (off by default)")
    p.add_argument("--loop-bars", type=int, default=4,
                   help="length of the fixed drum loop that's baked once and replayed")
    p.add_argument("--input-device", default="iD4",
                   help="DI input device: index or name substring")
    p.add_argument("--output-device", default=None,
                   help="output device: index or name substring (MUST differ from input)")
    p.add_argument("--input-channel", type=int, default=0)
    p.add_argument("--rms-gate", type=float, default=1e-3,
                   help="min RMS to attempt chord detection on a beat")
    p.add_argument("--modal-margin", type=float, default=0.6,
                   help="how much exotic modes must beat major/minor for the key estimate")
    p.add_argument("--note-threshold", type=float, default=0.3,
                   help="basic-pitch: min note-posterior to count a pitch")
    p.add_argument("--key-strength", type=float, default=0.0,
                   help="(currently unused: the backing follows the chord via the "
                        "prompt, not a notes mask) cfg_notes bias strength")
    p.add_argument("--style-strength", type=float, default=None, help="cfg_musiccoca")
    p.add_argument("--cfg-notes", type=float, default=None,
                   help="cfg_notes: how hard the baked backing's root-note hint mask "
                        "steers generation. Backing bake defaults to 1.0 (BACKING_CFG_NOTES); "
                        "lower it if the harmony feels overbearing.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--lead-seconds", type=float, default=1.5)
    p.add_argument("--backing-gain", type=float, default=0.9,
                   help="level of the pitched backing relative to drums when summed "
                        "(1.0 = equal; lower to keep the kit on top)")
    p.add_argument("--cfg-drums", type=float, default=3.0,
                   help="cfg_drums for the backing bake; pushes drums OFF harder so "
                        "the pitched layer doesn't compose its own kit (0 to relax)")
    p.add_argument("--mode", choices=(MODE_DRUMS, MODE_BACKING, MODE_MIXED),
                   default=MODE_MIXED,
                   help="initial playback layer (toggle live with d/b/m)")
    p.add_argument("--duplex", action="store_true",
                   help="use ONE device for both input and output via a single "
                        "full-duplex stream (e.g. the iD4) -- convenient, one cable. "
                        "Ignores --output-device.")
    p.add_argument("--backing-only", action="store_true",
                   help="ISOLATE the backing stage: skip the DI input, count-in, and "
                        "live learning entirely; take the progression from --chords and "
                        "play the baked backing SOLO (no drums). For iterating on the "
                        "backing generation without a guitar/learning pass.")
    p.add_argument("--chords", default="C, C, Dminor, C, E#, C",
                   help="--backing-only progression: comma-separated chords, one bar "
                        "each (e.g. 'C, Am, F, G'). Names are <letter><#/b?><quality>, "
                        "quality in maj/min/dim/aug spellings (empty = major).")
    p.add_argument("--no-repeat", action="store_true",
                   help="--backing-only only: play the baked progression ONCE and stop, "
                        "instead of looping forever. For hearing the backing without the "
                        "loop-point crossfade/repeat.")
    p.add_argument("--crossfade", action="store_true",
                   help="bake the backing WITH equal-power crossfades at chord changes, "
                        "long-chord repeats, and the loop point. Default is hard butt-splices "
                        "(seams sounded cleaner than the crossfade's smearing).")
    p.add_argument("--generation-debug", action="store_true",
                   help="debug mode (implies --backing-only): play each RAW generated clip "
                        "(one per chord span, exactly as the model produced it -- no tiling/"
                        "splicing/looping) ONCE, with 1s of silence between, then stop. For "
                        "hearing what the model actually generated per chord.")
    args = p.parse_args()
    if args.generation_debug:
        args.backing_only = True
    if args.no_repeat and not args.backing_only:
        p.error("--no-repeat only applies with --backing-only")

    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    bpb = _beats_per_bar(args.time_sig)

    # Backing bake params, computed up front (they don't depend on the progression) so the
    # progressive single-chord VAMP bake (Stage B, baked mid-learning) can use them too.
    # "Let the model breathe" knobs scoped to the BACKING bake only (the drum bake reads the
    # untouched args.* -> None -> model defaults). An explicit CLI flag still wins.
    band_prompt = prompt_with_tempo(args.prompt, args.tempo, args.time_sig)
    bk_style = args.style_strength if args.style_strength is not None else BACKING_CFG_MUSICCOCA
    bk_notes = args.cfg_notes if args.cfg_notes is not None else BACKING_CFG_NOTES
    bk_temp = args.temperature if args.temperature is not None else BACKING_TEMPERATURE
    bk_topk = args.top_k if args.top_k is not None else BACKING_TOP_K

    import sounddevice as sd

    # --backing-only: ISOLATE the backing stage. No DI input, no count-in, no live
    # learning -- the progression comes from --chords. We only need an OUTPUT device.
    if args.backing_only:
        capture = None
        duplex = None
        if args.output_device:
            out_idx = resolve_output_device(args.output_device)
        else:
            out_idx = sd.default.device[1]
        args.output_device = out_idx
        progression = parse_chords(args.chords, bpb)
        key = None
        print(f"[backing-only] progression from --chords: " + " | ".join(
            f"{chord_name(s['root'], s['quality'])}" for s in progression))

        # Backing SOLO: no drum loop is baked here -- we play just the pitched backing so
        # the harmony is heard naked. The player is configured below from the backing
        # buffer's own sample rate (Stage 3). `loop` is None: no drum layer.
        # EXCEPTION: --generation-debug also bakes the drum loop so its raw phrase can be
        # auditioned alongside the per-chord backing clips (prepended to the clip list).
        drum_clip = None
        if args.generation_debug:
            print(f"Baking {args.loop_bars}-bar drum loop (once)...")
            _ds, _dsr, _dch, _doff, drum_take = make_drum_loop(
                mrt, args.tempo, args.time_sig, args.loop_bars,
                cfg_musiccoca=args.style_strength, temperature=args.temperature,
                top_k=args.top_k, return_clip=True)
            drum_clip = ("drums", args.loop_bars * bpb, drum_take)
        loop = None
        player = None
    else:
        # We sample the input per beat as we go (not all at the end), so the ring
        # buffer only needs to hold a few seconds of trailing audio for one
        # _beat_chroma slice (~1.6s) plus margin. Two bars (min 3s) is ample.
        seconds_per_bar = (60.0 / args.tempo) * _beats_per_bar(args.time_sig)
        window = max(3.0, 2 * seconds_per_bar)
        capture = InputCapture(args.input_device, window, channel=args.input_channel,
                               rms_gate=args.rms_gate, modal_margin=args.modal_margin,
                               estimator="basic-pitch", note_threshold=args.note_threshold)

        # Device routing. Two modes:
        #  - duplex: ONE device, ONE full-duplex stream for in+out (the convenient iD4
        #    setup). Safe because it's a single PortAudio client on a single clock.
        #  - separate: distinct input/output devices, two streams. Two streams on ONE
        #    device collide (-10863), so we still forbid same-device in this mode.
        if args.duplex:
            out_idx = capture.idx
            args.output_device = out_idx
        else:
            if args.output_device:
                out_idx = resolve_output_device(args.output_device)
            else:
                out_idx = sd.default.device[1]
            if out_idx == capture.idx:
                raise SystemExit(
                    f"output device {sd.query_devices(out_idx)['name']!r} is the same as the DI "
                    f"input ({capture.name!r}); two separate PortAudio streams on one CoreAudio "
                    f"device collide (-10863). Either pass --duplex (one shared full-duplex "
                    f"stream on this device) or route output to a DIFFERENT device.")
            args.output_device = out_idx

        # The band prompt (args.prompt) is captured for the deferred chord-following
        # backing; the drum loop is baked from its own drums-only prompt (drum_prompt),
        # so the band words don't leak pitched instruments into the bare drum bed.
        print(f"DI input {capture.name!r} @ {capture.sr} Hz, channel {args.input_channel}; "
              f"{args.count_in_bars}-bar intro, then listening until the loop repeats.")

        # Bake the fixed drum loop ONCE, up front. Everything after (count-in, learning,
        # final playback) replays this exact buffer -- the model is never asked for drums
        # again, so the groove is steady and identical every bar.
        print(f"Baking {args.loop_bars}-bar drum loop (once)...")
        samples, sample_rate, channels, beat_offsets = make_drum_loop(
            mrt, args.tempo, args.time_sig, args.loop_bars,
            cfg_musiccoca=args.style_strength, temperature=args.temperature, top_k=args.top_k)

        player = QueuedPlayer(lead_seconds=args.lead_seconds,
                              output_device=args.output_device, external=args.duplex)
        player.configure(sample_rate, channels)

        duplex = None
        if args.duplex:
            # One shared stream at the model's rate (48k). The iD4 reports 44.1k as its
            # default, so align the capture's analysis rate to the stream (re-alloc its
            # ring buffer at 48k) -- otherwise basic-pitch would read the wrong pitches.
            # capture._stream (its own InputStream) is created-but-never-started, so it
            # holds no device reservation; the duplex stream owns the device.
            if capture.sr != sample_rate:
                print(f"  duplex: aligning input analysis rate {capture.sr} -> {sample_rate} Hz")
                capture.sr = sample_rate
                capture._alloc_buffer()
            duplex = DuplexStream(capture.idx, capture, player, sample_rate,
                                  capture.in_channels)
            print(f"Duplex on {capture.name!r}: input + output share one stream @ "
                  f"{sample_rate} Hz.")
            duplex.start()
        else:
            capture.start()

        # PROGRESSIVE learning (three stages):
        #  A: drums only (count-in + initial listening) -- the learner thread is the clock+ears.
        #  B: the instant bar 1's root is heard, bake a one-chord VAMP on THIS (main) thread
        #     and hand it to the learner, which layers it under the drums while STILL listening.
        #  C: when the loop locks, the learner keeps drums+vamp alive while we bake the full
        #     backing here, then we enter the full band at the top of the progression (below).
        # MLX stays on the main thread; the learner is copy-only (push + read-only snapshot).
        loop = DrumLoopFeeder(samples, beat_offsets, bpb)
        learner = ProgressionLearner(
            loop, capture, args.tempo, args.time_sig, player,
            count_in_bars=args.count_in_bars, beats_per_bar=bpb,
            backing_gain=args.backing_gain)
        learner.start()
        try:
            # Stage B (opt-in via --vamp): wait for the first chord, bake the single-chord
            # vamp, install it. (Skip in --generation-debug: it auditions raw clips and
            # returns; a vamp adds nothing.) Off by default -- many find the early single
            # chord muddies things until the full progression locks.
            if args.vamp and not args.generation_debug:
                first_root, first_quality = learner.wait_first_chord()
                vamp_prog = [{"start_beat": 0, "length_beats": bpb,
                              "root": first_root, "quality": first_quality}]
                print(f"Baking vamp on {chord_name(first_root, first_quality)} "
                      f"(progressive: comes in while we keep listening)...")
                v_samples, _v_sr, _v_ch, v_offsets = make_backing_loop(
                    mrt, band_prompt, args.tempo, args.time_sig, vamp_prog, None,
                    cfg_musiccoca=bk_style, cfg_notes=bk_notes,
                    cfg_drums=args.cfg_drums, temperature=bk_temp, top_k=bk_topk,
                    crossfade=args.crossfade)
                learner.install_vamp(DrumLoopFeeder(v_samples, v_offsets, bpb))

            # Stage C: block until the loop locks. The learner keeps drums+vamp going (and
            # holds lock_beat internally for its assumed-position cue + enter-at-top).
            progression, key, _lock_beat = learner.wait_locked()
        finally:
            if not args.duplex:
                capture.stop()  # done listening; separate streams: free the input device
            # In duplex the shared stream must keep running for playback; we keep
            # feeding its (now-ignored) input ring buffer, which is harmless.

    # Stage 3: play the pitched backing that follows the captured progression under
    # continuous drums forever. Two backing modes:
    #  - stream (default): generate ONE BAR at a time off MRT2, carrying model state
    # The backing is ALWAYS baked once up front (per-chord <=6s takes, looped), then
    # looped forever: make_backing_loop does one generate() per chord change capped at
    # ~6s (state carried, crossfade-joined, longer chords loop their take, seamless-
    # wrapped), so playback only starts after every span is generated and a fixed buffer
    # can't drift. Drums (when present) stay the fixed drums-only bed -- the master clock.
    # band_prompt + bk_* were computed up front (the vamp bake needs them).
    total_beats = sum(s["length_beats"] for s in progression)
    _bars = bar_chords(progression, total_beats, bpb)
    prog_bars = len(_bars)

    spans = chord_spans(progression, total_beats, bpb)
    print(f"Baking backing band over the progression ({prog_bars} bars, "
          f"{len(spans)} chord change(s), <={BACKING_MAX_TAKE_SECONDS:.0f}s/take)...")
    # The drums (+ the Stage-B vamp) keep sounding during this multi-second main-thread bake
    # because the learner thread is STILL the clock -- it stays alive from listening through
    # this bake to the top-of-progression entry below, printing the assumed live position each
    # bar. (In --backing-only there is no learner/clock; the bake runs synchronously and
    # silence-before-first-sound is fine.)
    baked = make_backing_loop(
        mrt, band_prompt, args.tempo, args.time_sig, progression, key,
        cfg_musiccoca=bk_style, cfg_notes=bk_notes,
        cfg_drums=args.cfg_drums, temperature=bk_temp, top_k=bk_topk,
        crossfade=args.crossfade, return_clips=args.generation_debug)

    if args.generation_debug:
        b_samples, b_sr, b_ch, b_offsets, clips = baked
        if drum_clip is not None:
            clips = [drum_clip] + clips  # audition the raw drum phrase before the backing clips
        player = QueuedPlayer(lead_seconds=args.lead_seconds,
                              output_device=args.output_device, external=False)
        label = (f"GENERATION DEBUG: raw drum loop + per-chord clips, {prog_bars}-bar progression "
                 f"@ {args.tempo} BPM {args.time_sig}")
        if not args.backing_only:
            learner.stop()  # live generation-debug ran the learner as the clock; stop it
        play_clips_with_silence(clips, b_sr, b_ch, label, player)
        if duplex is not None:
            duplex.close()
        return

    b_samples, b_sr, b_ch, b_offsets = baked
    backing = DrumLoopFeeder(b_samples, b_offsets, bpb)  # generic beat-slice feeder

    if args.backing_only:
        # No drums baked: configure the player from the backing buffer and play solo.
        player = QueuedPlayer(lead_seconds=args.lead_seconds,
                              output_device=args.output_device, external=False)
        player.configure(b_sr, b_ch)
        band = BackingSoloFeeder(backing, progression, total_beats, bpb,
                                 backing_gain=args.backing_gain)
        label = (f"Backing SOLO (no drums): {prog_bars}-bar progression "
                 f"@ {args.tempo} BPM {args.time_sig} (baked per-chord <=6s takes)")
    else:
        # Enter in phase with the live player: the learner is STILL the clock (drums + the
        # Stage-B vamp). Ask it to stop ON the next top-of-progression downbeat, so the vamp
        # keeps sounding right up to the swap (no drums-only gap) and loop.i is left sitting on
        # that downbeat -- the band's beat 0 (progression bar 1) then coincides with live bar 1.
        print("Waiting for the top of the progression to enter...")
        learner.enter_at_top()
        learner.wait_for_top()
        learner.stop()
        band = LayeredFeeder(loop, backing, progression, total_beats, bpb,
                             backing_gain=args.backing_gain, mode=args.mode)
        band.beat = 0  # progression bar 1, aligned to the live bar 1 we waited for
        loop_bars = len(beat_offsets) // bpb
        label = (f"Band: {loop_bars}-bar drum loop under a {prog_bars}-bar progression "
                 f"@ {args.tempo} BPM {args.time_sig} (baked per-chord <=6s backing)")
    try:
        if args.backing_only and args.no_repeat:
            play_band_once(band, label + " [play once, no repeat]", player)
        else:
            play_band_forever(band, label, player)
    finally:
        if duplex is not None:
            duplex.close()


if __name__ == "__main__":
    main()
