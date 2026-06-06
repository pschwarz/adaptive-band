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

Beat-synced live stream from a text prompt (Ctrl+C to stop):

```bash
uv run python hello_world.py --prompt "blues" --tempo 100 --time-sig 4/4
```

Args: `--prompt`, `--tempo` (BPM, default 100), `--time-sig` (`4/4`, `3/4`, or `6/8`,
default `4/4`), `--size` (default `mrt2_base`). Output is 48 kHz stereo to the speakers.
Key/scale control adds `--listen`, `--input-device`, `--analysis-bars`, `--key-strength`,
`--key`, `--mode` (see **Key / scale / mode** below).

### How it streams

Generation runs on the main thread (the imported `.mlxfn` is bound to the thread that loaded
it) and fills a queue; PortAudio's audio callback drains it. A ~1.5s lead buffer keeps the
device fed so it never starves between `generate()` calls — generation is ~0.6s per 1s of
audio on M4 Max, so playback is gapless.

### Beat sync

The stream is generated **one beat at a time** (each beat's length in 40ms frames derived
from `--tempo`), setting the model's `drums` conditioning to *play* only on **backbeat**
beats — 2 & 4 in `4/4`, 2 & 3 in `3/4`, beat 2 in `6/8`. Each backbeat is split into a
1-frame drum *onset* plus a tail, placing the hit at the start of the beat. A fractional-frame
accumulator carries the rounding remainder across beats, so the **average** tempo stays locked
even though each beat snaps to the 40ms grid (individual beats jitter ±~20ms).

This is the `drums` channel — a *soft* bias toward grid-aligned hits, not a sample-exact
metronome click, and the model may add its own off-grid percussion. It's the finest rhythmic
control the API exposes (MRT2 takes no audio/click input).

`--tempo` and `--time-sig` also feed a text hint into the prompt — e.g. `"blues, medium
tempo, 100 BPM, 4/4 time"` — since MRT2 has **no numeric tempo or time-signature input**; its
only conditioning is the style prompt (plus notes/drums). The prompt hint and the beat grid
reinforce each other.

### Key / scale / mode

MRT2 has **no symbolic key input** either. The real pitch lever is the `notes` conditioning
— a 128-int vector, one slot per MIDI pitch (0–127), each `-1` masked / `0` off / `1` on /
`2` onset / `3` on (model's choice). A *key* is a rule over pitch classes, so we expand a
`(root, mode)` into that vector: every in-scale pitch → `3`, every out-of-scale pitch → `-1`
(masked / "no opinion") — a **soft scale bias**.

> **Why `-1`, not `0`:** `0` means "this pitch is *off*" — a hard lock that forbids every
> out-of-scale pitch on *every* beat. Applied continuously with the streaming `state` carried
> forward, that fights the model's own evolving line and it **loses coherence within ~10 s**
> (even a static `--key`/`--mode`). `-1` ("no opinion") lets the model still pass through
> chromatic/leading tones; the bias toward the scale comes from the in-scale `3`s plus the
> text prompt, not from forbidding notes.

`--key-strength` sets `cfg_notes`, the guidance push on the whole `notes` conditioning. It
**defaults to `0`** (a gentle bias from the mask alone). Raising it pushes harder toward the
scale, but a strong *continuous* push degrades coherence over time the same way the hard lock
did — turn it up only a little, and only if you need a tighter key.

Two ways to choose the scale:

**Static** — pin a fixed key/mode for the whole stream (also reinforced in the text prompt):

```bash
uv run python hello_world.py --prompt "warm pads" --key C --mode dorian
```

`--key` takes `C`, `F#`, `Bb`, …; `--mode` is `ionian/dorian/phrygian/lydian/mixolydian/`
`aeolian/locrian` (or `major`/`minor`).

**Listen** — *follow what you play*. With `--listen`, a second PortAudio input stream
captures live audio (default device matches `iD4`; override with `--input-device <name|idx>`),
and at **each bar boundary** the last `--analysis-bars` (default 4) of audio are turned into a
chroma vector and matched against the 84 diatonic key templates (12 roots × 7 modes) to derive
`(root, mode)`. The new scale mask is swapped in on the next bar; the detected key is printed
only when it changes (`[key] C ionian`).

```bash
uv run python hello_world.py --prompt "jazz trio, piano" --listen
```

Notes:
- **Play into a line/instrument input** (the iD4 default), not a room mic — a mic would hear
  the model's own speaker output and the key estimate would chase itself. Use headphones if
  you must use a mic.
- **Mode detection is inherently fragile**: relative modes (e.g. A aeolian vs C ionian) share
  the same seven notes, and neighbors like dorian/aeolian differ by a single colour tone (the
  6th). The estimator weights tonic + fifth to break the tie, and the wide analysis window
  stabilizes it, but if you don't actually *sound* the characteristic tone it may land on a
  neighboring mode. To keep this tame, the estimator **defaults to major/minor** and only
  reports an exotic mode (dorian, lydian, …) when the chroma beats the best major/minor by
  `--modal-margin` (default 0.35); set it to `0` for raw best-fit, or higher to stick harder
  to major/minor. So you only get dorian when you really play the natural 6th, etc.
- A silent/near-silent bar keeps the previous mask (the scale doesn't lurch to noise).

The input callback only copies samples (no MLX) — generation stays on the main thread, same
discipline as the output path.

`--listen` runs **two independent PortAudio streams**: an `InputStream` on the listen device
(key detection) and an `OutputStream` for generation. They must be **different devices** —
two streams on one CoreAudio device fail with `-10863` ("cannot do in current context"), and a
single *duplex* stream would slave the output to the input's clock, drifting the beat grid over
a long run (the drums lose the beat first). Independent streams keep generation on its own clock
so the backbeat stays locked.

So when you listen on the iD4, send output somewhere else: set the **macOS system output** to
e.g. *External Headphones* (the Mac's own jack), or pass `--output-device <name|idx>`. If the
resolved output equals the listen input, the run aborts with a message telling you to reroute —
it won't silently collide or drift.

#### Estimators

`--listen` has two front-ends that produce the 12-class pitch evidence feeding the key
estimator (both then share the same weighted-template + modal-margin logic):

- `--estimator chroma` (default) — librosa `chroma_cqt`. Fast, no extra model, but it's
  smeared energy: overtones and bin bleed can hide the characteristic tone, which is the
  main source of mode mistakes.
- `--estimator basic-pitch` — Spotify's [basic-pitch](https://github.com/spotify/basic-pitch)
  polyphonic note model. It detects actual notes, so the pitch-class histogram is sharper.
  The model's input is a fixed ~2 s tensor, but we **tile** it across the whole analysis
  window (summing the per-chunk note histograms), so single-note playing — e.g. a bass
  outlining a chord note by note — accumulates into one clean key over the full window.
  Heavier per bar but you're latency-tolerant here (runs on the main thread, where chroma
  already does). Tune `--note-threshold` (default 0.3) by ear: raise it if it reports phantom
  notes, lower it if it misses quiet ones.

Window length: by default the analysis window is `--analysis-bars` bars wide; set
`--listen-seconds N` to pin it to N seconds directly (0 = derive from bars). A longer window
(e.g. `--listen-seconds 6`) helps single-note/bass playing accumulate enough notes to settle
a key. `--monitor` honors `--estimator`, `--note-threshold`, and `--listen-seconds` too, so
you can dial the window in diagnostically before running the full stream.

We run basic-pitch via its **bundled ONNX model** (vendored at `models/nmp.onnx`, 228 KB)
through `onnxruntime` — *not* the `basic-pitch` pip package, which pulls in TensorFlow and
pins `numpy<2` and so breaks MRT2/MLX (which needs numpy 2). Only `onnxruntime` is added.
