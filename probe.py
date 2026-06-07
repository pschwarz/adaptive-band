"""Direct-to-Magenta prompt probe: a tight loop for testing prompts.

This is a DELIBERATELY minimal entry point, separate from staged_band.py. It
strips away the whole staged pipeline (drums bed -> learn progression -> bake
backing -> loop) and goes straight at MRT2's one real lever -- the text style
prompt -- so you can iterate on prompt wording, length, and the generation knobs
and just listen.

Run it (each call generates ONE clip and, by default, plays it once + writes a WAV):
  uv run python probe.py --prompt "lo-fi hip hop, dusty rhodes" --bars 8 --tempo 86
  uv run python probe.py --prompt "solo upright bass walking line" --bars 4 --temperature 1.2
  uv run python probe.py --prompt "warm pad" --bars 8 --drums on        # let it add a kit
  uv run python probe.py --prompt "..." --no-play --out takes/idea.wav   # just write a file

What it gives you (same params as staged_band where they matter):
  --prompt       free-form style text -- the thing you're here to play with.
  --tempo        BPM. MRT2 has no numeric tempo input, so this drives the beat
                 grid (how many frames to ask for) AND is appended to the prompt
                 as a tempo word + "N BPM" hint (prompt_with_tempo), exactly like
                 the other tools.
  --time-sig     4/4 | 3/4 | 6/8 -- drives the beat grid and a soft prompt hint.
  --bars         how many bars to generate (length). One generate() call, like
                 staged_band's make_backing_loop bakes a whole phrase at once.
  --drums        off (default) / on / auto -- MRT2's drums conditioning int
                 (0 off / 1 on / -1 masked-"auto"). staged_band found this is a
                 real lever: "off" stops a pitched prompt composing its own kit;
                 "on" forces percussion in; "auto" lets the model decide.
  --temperature / --top-k / --style-strength  the sampling + cfg_musiccoca knobs.

No looping, no crossfade-wrap, no input capture -- just text in, audio out, so the
sound you hear is exactly what the prompt produced (no seam-hiding, no layering).
"""

import argparse

from hello_world import (
    BEAT_GRID,
    MagentaRT2SystemMlxfn,
    prompt_with_tempo,
    resolve_output_device,
)
# Reuse staged_band's deterministic beat-frame planner so a requested bar count maps
# to the same frame budget the rest of the project uses (no duplicated grid math).
from staged_band import _beat_frame_counts, _beats_per_bar


# MRT2's drums conditioning is a single int per generate() call.
DRUMS = {"off": 0, "on": 1, "auto": -1}


def generate_clip(mrt, prompt, bpm, time_sig, bars, drums,
                  cfg_musiccoca, temperature, top_k):
    """Generate `bars` bars from `prompt` in ONE generate() call. Returns
    (samples float32 (N,ch), sample_rate, channels). No wrap, no trim -- the raw
    model output for exactly the planned frame budget."""
    import numpy as np

    bpb = _beats_per_bar(time_sig)
    beat_frames = _beat_frame_counts(bpm, time_sig, bars * bpb)
    gen_frames = sum(beat_frames)

    embedding = mrt.embed_style(prompt)
    wav, _state = mrt.generate(
        style=embedding, frames=gen_frames, state=None,
        drums=[DRUMS[drums]], notes=None, cfg_musiccoca=cfg_musiccoca,
        temperature=temperature, top_k=top_k)
    samples = np.ascontiguousarray(wav.samples, dtype=np.float32)
    return samples, wav.sample_rate, wav.num_channels


def play(samples, sample_rate, channels, output_device=None):
    """Play `samples` once, blocking until done."""
    import sounddevice as sd

    out_idx = resolve_output_device(output_device) if output_device else None
    sd.play(samples, samplerate=sample_rate, device=out_idx)
    sd.wait()


def write_wav(path, samples, sample_rate):
    import os

    import soundfile as sf

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    sf.write(path, samples, sample_rate)


def main():
    p = argparse.ArgumentParser(
        description="Direct-to-Magenta prompt probe (one clip per run).")
    p.add_argument("--prompt", required=True,
                   help="free-form style prompt to test")
    p.add_argument("--tempo", type=float, default=100.0,
                   help="BPM: drives the beat grid + a soft prompt hint")
    p.add_argument("--time-sig", choices=tuple(BEAT_GRID), default="4/4",
                   help="time signature: drives the beat grid + a soft prompt hint")
    p.add_argument("--bars", type=int, default=8,
                   help="how many bars to generate (one generate() call)")
    p.add_argument("--drums", choices=tuple(DRUMS), default="off",
                   help="MRT2 drums conditioning: off / on / auto (model decides)")
    p.add_argument("--size", default="mrt2_base")
    p.add_argument("--style-strength", type=float, default=None,
                   help="cfg_musiccoca (how hard to push toward the prompt)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="WAV path to write (default: auto-named in takes/)")
    p.add_argument("--no-write", action="store_true", help="don't write a WAV")
    p.add_argument("--no-play", action="store_true", help="don't play the clip")
    p.add_argument("--output-device", default=None,
                   help="playback device: index or name substring")
    args = p.parse_args()

    prompt = prompt_with_tempo(args.prompt, args.tempo, args.time_sig)
    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    print(f"Generating {args.bars} bars @ {args.tempo} BPM {args.time_sig}, "
          f"drums={args.drums}")
    print(f"  prompt: {prompt!r}")
    samples, sample_rate, channels = generate_clip(
        mrt, prompt, args.tempo, args.time_sig, args.bars, args.drums,
        cfg_musiccoca=args.style_strength, temperature=args.temperature,
        top_k=args.top_k)
    dur = samples.shape[0] / sample_rate
    print(f"  -> {dur:.1f}s, {sample_rate} Hz, {channels} ch")

    if not args.no_write:
        out = args.out
        if out is None:
            import re
            import time
            slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
            out = f"takes/{int(time.time())}_{slug}.wav"
        write_wav(out, samples, sample_rate)
        print(f"  wrote {out}")

    if not args.no_play:
        print("  playing...")
        play(samples, sample_rate, channels, args.output_device)
    print("Done.")


if __name__ == "__main__":
    main()
