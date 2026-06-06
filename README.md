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

## Run it

Beat-synced live stream from a text prompt (Ctrl+C to stop):

```bash
uv run python hello_world.py --prompt "blues" --tempo 100 --time-sig 4/4
```

Args: `--prompt`, `--tempo` (BPM, default 100), `--time-sig` (`4/4`, `3/4`, or `6/8`,
default `4/4`), `--size` (default `mrt2_base`). Output is 48 kHz stereo to the speakers.

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
