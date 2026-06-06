"""Minimal Magenta RealTime 2 hello-world: text prompt -> audio file.

Uses MagentaRT2SystemMlxfn so it loads the already-downloaded .mlxfn weights
(from ~/Documents/Magenta/magenta-rt-v2) with zero network access.\

Run it / listen:
uv run python hello_world.py --prompt "disco funk" --seconds 4 --out out.wav
afplay out.wav

"""
import argparse
from magenta_rt.mlx.system import MagentaRT2SystemMlxfn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="warm ambient synth pads")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--size", default="mrt2_base")  # 2.4B: full quality, real-time on M4 Max
    p.add_argument("--out", default="out.wav")
    args = p.parse_args()

    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    print(f"Embedding prompt: {args.prompt!r}")
    embedding = mrt.embed_style(args.prompt)

    frames = round(args.seconds * 25)  # 25 frames == 1 second
    print(f"Generating ~{args.seconds}s ({frames} frames)...")
    wav, _state = mrt.generate(style=embedding, frames=frames)

    wav.write(args.out)
    print(f"Wrote {args.out} @ {wav.sample_rate} Hz")


if __name__ == "__main__":
    main()
