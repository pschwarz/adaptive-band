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

Uses `MagentaRT2SystemMlxfn`, which loads the already-exported `.mlxfn` weights with no
network access. Output is 48 kHz stereo WAV.
