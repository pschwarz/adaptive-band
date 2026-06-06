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
```

## Hello world

Generate audio from a text prompt:

```bash
uv run python hello_world.py --prompt "disco funk" --seconds 4 --out out.wav
afplay out.wav
```

Or play live to the speakers:

```bash
uv run python hello_world.py --prompt "disco funk" --seconds 4 --play   # fixed-length
uv run python hello_world.py --prompt "disco funk" --stream             # endless, Ctrl+C to stop
```

`--stream` generates ~1s chunks back-to-back, threading the model's state forward so the
music stays coherent. Generation runs on the main thread (the imported `.mlxfn` is bound to
the thread that loaded it) and fills a small bounded queue; PortAudio's audio callback
drains it. The few-chunk lead buffer keeps the device fed so it never starves between
`generate()` calls — generation is ~0.6s per 1s of audio on M4 Max, so playback is gapless.
(Serializing generate → write instead caused an audible seam every chunk.)

### Tempo & time signature

`--tempo` (BPM, default 100) and `--time-sig` (`4/4`, `3/4`, or `6/8`) nudge the rhythm:

```bash
uv run python hello_world.py --prompt "waltz" --tempo 120 --time-sig 3/4 --stream
```

MRT2 has **no numeric tempo or time-signature input** — its only conditioning is the text
style prompt (plus notes/drums). So both are *soft hints*: they append a tempo word + BPM
(and the time signature, if given) to the prompt — e.g. `"waltz, upbeat tempo, 120 BPM,
3/4 time"` — before embedding. They influence feel but do not lock tempo or meter.

### Beat-synced drums

`--beat-sync` drives an actual rhythm instead of just hinting one. It generates **one beat
at a time** (deriving each beat's length in 40ms frames from `--tempo`) and sets the model's
`drums` conditioning to *play* only on **backbeat** beats — 2 & 4 in `4/4`, 2 & 3 in `3/4`,
beat 2 in `6/8`. Implies streaming; Ctrl+C to stop.

```bash
uv run python hello_world.py --prompt "funk" --tempo 100 --time-sig 4/4 --beat-sync
```

Each backbeat is split into a 1-frame drum *onset* plus a tail, placing the hit at the start
of the beat. A fractional-frame accumulator carries the rounding remainder across beats, so
the **average** tempo stays locked even though each beat snaps to the 40ms grid (individual
beats jitter ±~20ms). This is still the `drums` channel — a *soft* bias toward grid-aligned
hits, not a sample-exact metronome click, and the model may add its own off-grid percussion.
It's the finest rhythmic control the API exposes (MRT2 takes no audio/click input).

Uses `MagentaRT2SystemMlxfn`, which loads the already-exported `.mlxfn` weights with no
network access. Output is 48 kHz stereo WAV.
