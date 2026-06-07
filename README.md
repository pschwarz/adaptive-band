# adaptive-band

Experiments with [Magenta RealTime 2](https://magenta.withgoogle.com/magenta-realtime-2)
(MRT2) — Google's open-weights live music model — running natively on Apple Silicon via MLX.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Apple Silicon Mac. Model weights are
expected at `~/Documents/Magenta/magenta-rt-v2/` (downloaded by the MRT2 desktop/AU apps;
otherwise `mrt models download`).

```bash
uv venv --python 3.12
uv pip install "magenta-rt[mlx]"
uv pip install onnxruntime   # for --estimator basic-pitch (see Key / scale / mode)
```

## Run it

The entry point is **`staged_band.py`** — a staged backing band that follows a chord
progression you play. Full walkthrough in **[Staged backing band](#staged-backing-band-staged_bandpy)**
below; the quick start:

```bash
uv run python staged_band.py --prompt "jazz trio, piano" --tempo 100 --time-sig 4/4 \
    --input-device iD4 --output-device "External Headphones"
```

`mrt_core.py` is the supporting **library** (no CLI): the MRT2 loader, prompt/tempo hints, the
beat grid, live input capture, the key/chord estimators, and device resolvers that
`staged_band.py` (and the `probe*.py` scratch scripts) import. The sections below explain the
core MRT2 levers those pieces are built on.

### How generation is driven

Generation runs on the main thread (the imported `.mlxfn` is bound to the thread that loaded
it) and fills a queue; PortAudio's audio callback drains it (`QueuedPlayer`). A ~1.5s lead
buffer keeps the device fed so it never starves between `generate()` calls — generation is
~0.6s per 1s of audio on M4 Max, so playback is gapless.

### Tempo & time signature

MRT2 has **no numeric tempo or time-signature input**; its only conditioning is the style
prompt (plus notes/drums). So `--tempo`/`--time-sig` do two things: they drive the **beat
grid** (each beat's length in 40ms frames; backbeats are 2 & 4 in `4/4`, 2 & 3 in `3/4`, beat 2
in `6/8`), and they feed a text hint into the prompt — e.g. `"jazz trio, medium tempo, 100 BPM,
4/4 time"`. A fractional-frame accumulator carries the rounding remainder across beats, so the
**average** tempo stays locked even though each beat snaps to the 40ms grid.

The `drums` channel is the finest rhythmic control the API exposes — a *soft* bias toward
grid-aligned hits on backbeats, not a sample-exact metronome click (MRT2 takes no audio/click
input), so the model may add its own off-grid percussion.

### Key / scale / mode

MRT2 has **no symbolic key input** either. The real pitch lever is the `notes` conditioning
— a 128-int vector, one slot per MIDI pitch (0–127), each `-1` masked / `0` off / `1` on /
`2` onset / `3` on (model's choice). A *key* is a rule over pitch classes, so `scale_mask`
expands a `(root, mode)` into that vector: every in-scale pitch → `3`, every out-of-scale
pitch → `-1` (masked / "no opinion") — a **soft scale bias**. (The backing band uses a gentler
single-root variant of this; see below.)

> **Why `-1`, not `0`:** `0` means "this pitch is *off*" — a hard lock that forbids every
> out-of-scale pitch on *every* beat. Applied continuously with the streaming `state` carried
> forward, that fights the model's own evolving line and it **loses coherence within ~10 s**.
> `-1` ("no opinion") lets the model still pass through chromatic/leading tones; the bias
> toward the scale comes from the in-scale `3`s plus the text prompt, not from forbidding notes.

`cfg_notes` is the guidance push on the whole `notes` conditioning. A gentle bias from the mask
alone keeps the model coherent; raising it pushes harder toward the scale, but a strong
*continuous* push degrades coherence over time the same way the hard lock did.

#### Detecting the key/chords from live input

When `staged_band.py` listens to your DI instrument, a second PortAudio `InputStream`
(`InputCapture`) captures live audio into a rolling ring buffer. Each window of audio is turned
into 12-class pitch evidence and matched against weighted diatonic templates (12 roots × 7
modes) to derive `(root, mode)` (`estimate_key`); chord roots are matched the same way
(`chord_templates` / `_bar_root`).

Notes:
- **Play into a line/instrument input** (the iD4 default), not a room mic — a mic would hear
  the model's own speaker output and the estimate would chase itself. Use headphones if you
  must use a mic.
- **Mode detection is inherently fragile**: relative modes (e.g. A aeolian vs C ionian) share
  the same seven notes, and neighbors like dorian/aeolian differ by a single colour tone (the
  6th). The estimator weights tonic + fifth to break the tie, and a wide analysis window
  stabilizes it, but if you don't actually *sound* the characteristic tone it may land on a
  neighboring mode. To keep this tame, the estimator **defaults to major/minor** and only
  reports an exotic mode (dorian, lydian, …) when the chroma beats the best major/minor by
  `--modal-margin` (default 0.6); set it to `0` for raw best-fit, or higher to stick harder
  to major/minor.
- A silent/near-silent window keeps the previous estimate (the key doesn't lurch to noise).

The input callback only copies samples (no MLX) — generation stays on the main thread, same
discipline as the output path.

#### Estimators

Two front-ends produce the 12-class pitch evidence feeding the estimator (both then share the
same weighted-template + modal-margin logic):

- **chroma** (default) — librosa `chroma_cqt`. Fast, no extra model, but it's smeared energy:
  overtones and bin bleed can hide the characteristic tone, the main source of mode mistakes.
- **basic-pitch** — Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) polyphonic
  note model (`basic_pitch_histogram`). It detects actual notes, so the pitch-class histogram
  is sharper. The model's input is a fixed ~2 s tensor, but we **tile** it across the whole
  analysis window (summing the per-chunk note histograms), so single-note playing — e.g. a
  bass outlining a chord note by note — accumulates into one clean key over the full window.
  Tune `--note-threshold` (default 0.3) by ear: raise it if it reports phantom notes, lower it
  if it misses quiet ones.

We run basic-pitch via its **bundled ONNX model** (vendored at `models/nmp.onnx`, 228 KB)
through `onnxruntime` — *not* the `basic-pitch` pip package, which pulls in TensorFlow and
pins `numpy<2` and so breaks MRT2/MLX (which needs numpy 2). Only `onnxruntime` is added.

## Staged backing band (`staged_band.py`)

The band runs in stages and follows a **chord progression** you play, rather than just tracking
a key:

1. **Drums + tempo** — a fixed drum loop is **baked once** (`--loop-bars`, default 4) and
   then **replayed** for the whole session; it never drops out and never drifts.
2. **Learning stage** — after a count-in (`--count-in-bars`, default 4), it listens to your
   directly-wired (DI) instrument for `--learn-bars` bars (default 16) and captures the chord
   progression you play.
3. **Loop** — it bakes a pitched backing over the captured progression, then loops it
   **summed with the drums** forever under Ctrl+C, with the detected key/progression printed.

```bash
uv run python staged_band.py --prompt "jazz trio, piano" --tempo 100 --time-sig 4/4 \
    --input-device iD4 --output-device "External Headphones"
```

Console output walks the stages: `Baking 4-bar drum loop…`, `count-in…`, `intro bar 1/4 … 4/4`,
`LEARNING bar 1/16 … 16/16`, then the detected `[key]` and `[progression]`, then
`Baking backing band over the progression…`, then the looping band (drums + backing).

**Why a baked loop.** MRT2 composes percussion *fresh on every `generate()` call*, so streaming
drums beat-by-beat makes the groove wander and drift. Instead we ask the model for a short
drum-only passage **once** (`make_drum_loop`, `--loop-bars` bars), keep the resulting samples,
and **replay that exact buffer** (`DrumLoopFeeder`) for the count-in, the learning stage, and the
forever-loop. No model calls after baking — the beat is rock-steady and identical every bar.

**Drums-only.** The bake does *not* use the band prompt. MRT2's only real lever is the text
style, and with `notes=None` alone a band prompt (e.g. "indie rock ballad") still renders the
full kit-plus-band — piano/bass/guitar leak into the bed. So the loop is embedded from a
dedicated, emphatic drums-only style (`drum_prompt`: "solo drum kit, drums only … no bass, no
guitar, no piano … isolated drum track") with the tempo/time-sig hint appended. The band prompt
is kept only for the deferred chord-following backing.

**Seamless wrap.** Baked beat-by-beat with `state` carried forward, the buffer's end never
anticipates jumping back to sample 0, so a raw wrap pops once per loop. The bake generates **one
extra bar past the loop** (a lead-out — the model's natural continuation) and equal-power
crossfades it back over the loop's head (`_crossfade_wrap`), splicing the loop's end onto its
start with matching energy. The crossfade touches only the first beat, so the per-beat offsets
stay exact.

**Following chords.** After learning, a pitched backing is **baked once** over the captured
progression (`make_backing_loop`) and looped **summed with the drums** (`LayeredFeeder`). MRT2 has
no symbolic chord input. What shapes the sound:

- **Per-chord-change takes, ≤6s each, state carried.** We've stopped reasoning bar-by-bar —
  only chord **changes** matter. The progression is collapsed into chord **spans** measured in
  beats (`chord_spans`), and each span gets **one `generate()` call** of `min(span, ~6s)`
  (`BACKING_MAX_TAKE_SECONDS`), with MRT `state` carried span→span. A span that fits in ≤6s is
  generated at its exact length; a chord held **longer** than 6s generates a single ~6s take and
  **loops it** (crossfaded) to fill the span, rather than generating a long open-loop take that
  drifts. Each take **starts on its chord-change beat boundary**; its internal pulse is left to
  the model (tempo + time-sig in the prompt) — we don't re-generate or hard-align per beat.
  Span→span seams are equal-power crossfaded (`_crossfade_join`); a long span's internal repeat
  and the whole-loop point are both folded with a lead-out (`_crossfade_wrap`) — click-free.
- **Root-only detection.** The backing only cares about each chord's **root**. Quality is
  normalized to major in `chord_spans`, so a G major → G minor change is ignored (both sent as
  plain "G") and same-root beats merge regardless of quality. This deliberately smooths the
  backing for consistency — the kept harmony is just the span's root.
- **Root in the prompt + a single-pitch notes mask.** The span's prompt names the root
  (`bar_prompt` → `"<band prompt>, current chord: G major"`) and `notes = root_hint_mask(root)`
  marks the **one** root pitch class (`on=3`). The old graded chord+key mask (`chord_mask`,
  kept for reference) sounded bad; a lone root hint is a gentle cage, not a chord voicing.
- **Magenta reference params, backing-only.** The backing bake runs with Magenta's reference
  conditioning — `cfg_musiccoca=3.0`, `cfg_notes=1.0`, `temperature=1.3`, `top_k=40` (the
  `BACKING_*` constants). An earlier "let the model breathe" set (faint prompt + wide top-k)
  sounded worse than the library examples; an A/B confirmed the reference params. These apply
  to the **backing bake only**; the drum bake keeps model defaults. Any explicit CLI flag
  (`--style-strength`, `--cfg-notes`, `--temperature`, `--top-k`) overrides the backing default.
- **`drums=0` (off) on the backing, not `None`.** MRT2's `drums` conditioning is one int —
  `-1` masked / `0` off / `1` on — and `None` means *masked* = "model's choice", so the backing
  layer was composing its **own** kit that clashed with the drum bed. The backing is baked with
  `drums=0` (pushed by `--cfg-drums`, default 3) so it's strictly pitched.

The drums (the bed) and the backing wrap at their own lengths and are mixed per beat
(`--backing-gain`, soft-limited so the sum can't clip). The drum↔chord phase rotates as the two
loops wrap independently — accepted.

**No silence during the bake.** The backing bake is multi-second and runs on the main thread.
To avoid dropping to silence between learning and play, a background thread (`DrumKeepAlive`)
keeps pushing the baked drum loop into the player's queue while the bake runs, then stops and
joins cleanly before the real feeder takes over (copy-only, no MLX off the main thread).

**Iterate the backing without a guitar.** `--backing-only` skips the DI input, count-in, and
learning entirely: it takes the progression from `--chords` (comma-separated, one bar each, e.g.
`--chords "Am, F, C, G"`) and plays the baked backing **solo** (no drums). For tuning the
backing generation on its own — only needs an output device.

**Live debug hotkeys.** During the final loop: **`d`** = drums only, **`b`** = backing only,
**`m`** = mixed (default; set the initial layer with `--mode`), **`q`** = quit. The chord of each
bar is printed as it starts (`bar 3: Am`).

**Chord detection (v1).** Fully local, no network: each captured beat's pitches (via the
basic-pitch ONNX front-end) are folded into a 12-class histogram and matched against the triad
templates (`chord_templates`) for the strongest **root** (`_bar_root` — quality is ignored, the
backing keys off the root), adjacent identical roots are merged into segments, and the key is
estimated once over the whole window. The per-beat note table is built so it can later be sent
to an LLM (Sonnet) for richer labeling (sevenths, inversions) by swapping only the matching step.

`staged_band.py` imports its beat grid, input capture, pitch front-end, key/chord estimators,
and device resolvers from the `mrt_core` library.

**Device routing.** Two options:
- **`--duplex`** (convenient) — input and output share **one device** (e.g. the iD4) on a single
  full-duplex `sd.Stream`. This is safe because it's *one* PortAudio client on *one* clock: the
  `-10863` collision only happens with two *separate* streams on a device, and there's nothing to
  drift against with a single clock. The interface defaults to 44.1k but MRT2 is 48k, so the
  stream opens at 48k and the input analysis rate is realigned to match. One cable, no rerouting.
- **separate devices** (default) — distinct input and output devices, two streams. Same-device is
  still rejected here (two streams collide); pass `--duplex` or route output elsewhere.

Args: `--prompt`, `--tempo`, `--time-sig`, `--size`, `--loop-bars`, `--count-in-bars`,
`--learn-bars`, `--input-device`, `--output-device`, `--duplex`, `--input-channel`, `--rms-gate`,
`--modal-margin`, `--note-threshold`, `--key-strength` (now unused), `--style-strength`,
`--cfg-notes`, `--temperature`, `--top-k`, `--lead-seconds`, `--backing-gain`, `--cfg-drums`,
`--mode`, `--backing-only`, `--chords`.

Future directions (not yet built): continuous re-learning (regenerate the backing when a new
progression is detected) and gesture cues (a camera "nod" → switch progression for verse/chorus).
