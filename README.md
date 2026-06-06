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

Uses `MagentaRT2SystemMlxfn`, which loads the already-exported `.mlxfn` weights with no
network access. Output is 48 kHz stereo WAV.
