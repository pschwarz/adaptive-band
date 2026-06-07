"""Measure MRT2 generation speed vs real-time for a single bar, per model size.

No audio device, no input -- just times generate() so we can see whether the
streaming backing can keep up. A bar at 100 BPM 4/4 is 60 frames / 2.4s wall-clock;
if generating those 60 frames takes longer than 2.4s, the streaming backing can't
keep a lead and will continuously underrun -- the fix is a faster size (mrt2_small)
or a slower/structurally-cheaper setup, not more buffering.

Run: .venv/bin/python probe_rtf.py [--sizes mrt2_small mrt2_base] [--tempo 100]
"""
import argparse
import time

from hello_world import MagentaRT2SystemMlxfn
from staged_band import _beat_frame_counts, _beats_per_bar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["mrt2_small", "mrt2_base"])
    p.add_argument("--tempo", type=float, default=100.0)
    p.add_argument("--time-sig", default="4/4")
    p.add_argument("--bars", type=int, default=3, help="bars to time (skips warmup bar)")
    args = p.parse_args()

    bpb = _beats_per_bar(args.time_sig)
    frames = sum(_beat_frame_counts(args.tempo, args.time_sig, bpb))
    bar_wall = 60.0 / args.tempo * bpb
    print(f"One bar @ {args.tempo} BPM {args.time_sig} = {frames} frames = "
          f"{bar_wall:.2f}s wall-clock\n")

    for size in args.sizes:
        print(f"=== {size} ===")
        try:
            mrt = MagentaRT2SystemMlxfn(size=size)
        except Exception as e:
            print(f"  load failed: {e!r}\n")
            continue
        emb = mrt.embed_style("warm ambient synth pads, current chord: C major")
        state = None
        times = []
        for i in range(args.bars + 1):  # +1 warmup (first call includes JIT/compile)
            t0 = time.time()
            wav, state = mrt.generate(style=emb, frames=frames, state=state,
                                      drums=[0], notes=None)
            dt = time.time() - t0
            tag = " (warmup, ignored)" if i == 0 else ""
            print(f"  bar {i}: {dt:.2f}s  RTF {dt / bar_wall:.2f}{tag}")
            if i > 0:
                times.append(dt)
        avg = sum(times) / len(times)
        rtf = avg / bar_wall
        verdict = "OK (can keep up)" if rtf < 0.85 else (
            "TIGHT (little/no lead)" if rtf < 1.0 else "TOO SLOW (will underrun)")
        print(f"  avg {avg:.2f}s  RTF {rtf:.2f}  -> {verdict}\n")


if __name__ == "__main__":
    main()
