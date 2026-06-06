"""Magenta RealTime 2 beat-synced live stream: text prompt -> endless audio.

Uses MagentaRT2SystemMlxfn so it loads the already-downloaded .mlxfn weights
(from ~/Documents/Magenta/magenta-rt-v2) with zero network access.

Run it (Ctrl+C to stop):
uv run python hello_world.py --prompt "blues" --tempo 100 --time-sig 4/4
"""
import argparse
import time
from magenta_rt.mlx.system import MagentaRT2SystemMlxfn


def tempo_word(bpm):
    """Coarse tempo descriptor for the style prompt."""
    if bpm <= 70:
        return "slow tempo"
    if bpm <= 110:
        return "medium tempo"
    if bpm <= 140:
        return "upbeat tempo"
    return "fast tempo"


TIME_SIGS = ("4/4", "3/4", "6/8")


def prompt_with_tempo(prompt, bpm, time_sig=None):
    """MRT2 has no numeric tempo or time-signature input; the only lever is the
    text style prompt. Inject a tempo word + BPM (and time signature, if given)
    as a soft hint to MusicCoCa."""
    n = int(bpm) if float(bpm).is_integer() else bpm  # "100 BPM" not "100.0 BPM"
    hint = f"{prompt}, {tempo_word(bpm)}, {n} BPM"
    if time_sig:
        hint += f", {time_sig} time"
    return hint


# beats-per-bar and the (1-based) beats that get the backbeat drum tag.
BEAT_GRID = {
    "4/4": (4, {2, 4}),
    "3/4": (3, {2, 3}),
    "6/8": (2, {2}),  # compound: 2 dotted-quarter beats, backbeat on beat 2
}


def beat_plan(bpm, time_sig):
    """Yield (frames, drums) per generate() call, one beat at a time, forever.

    A backbeat beat is split into a 1-frame onset (drums=[1]) + an (N-1)-frame
    tail (drums=None) so the hit lands at the beat's start; other beats are a
    single N-frame call (drums=None). A fractional-frame accumulator carries the
    remainder across beats so the average tempo stays locked despite the 40ms
    (1-frame) grid — individual beats jitter +-20ms but the grid doesn't drift.
    """
    beats_per_bar, backbeats = BEAT_GRID[time_sig]
    # 6/8 beat = dotted quarter = 1.5x a quarter note.
    quarters_per_beat = 1.5 if time_sig == "6/8" else 1.0
    spb = (60.0 / bpm) * 25.0 * quarters_per_beat  # exact frames per beat
    acc = 0.0
    beat = 0
    while True:
        beat = beat % beats_per_bar + 1
        acc += spb
        n = round(acc)
        acc -= n
        if n < 1:
            n = 1
        if beat in backbeats and n >= 2:
            yield (1, [1])        # onset
            yield (n - 1, None)   # tail
        else:
            yield (n, [1] if beat in backbeats else None)


def stream_beats(mrt, embedding, bpm, time_sig, lead_seconds=1.5):
    """Beat-synced streaming: generate one beat at a time (see beat_plan), tagging
    the drum conditioning on backbeats, and stream gaplessly until Ctrl+C.

    Generation must stay on this (main) thread — the imported .mlxfn function is
    bound to the thread that imported the model, so calling generate() elsewhere
    raises "no Stream(gpu, N) in current thread". So generation runs here and
    fills a queue; PortAudio's own audio thread runs the callback, which only does
    buffer copies (no MLX). The lead cushion keeps the device from starving between
    generate() calls. Generation is ~0.6s per 1s of audio, so we stay ahead.

    The producer is driven by beat_plan, so queued slices are variable length; the
    lead cushion is therefore sized by audio seconds, not chunk count.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    plan = beat_plan(bpm, time_sig)

    # Prime first slice to learn sample rate + channel count.
    frames, drums = next(plan)
    wav, state = mrt.generate(style=embedding, frames=frames, drums=drums)
    sample_rate, channels = wav.sample_rate, wav.num_channels
    lead_samples = int(lead_seconds * sample_rate)

    chunks: queue.Queue = queue.Queue(maxsize=64)  # generous; backpressure via lead loop below
    chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))

    carry = np.empty((0, channels), dtype=np.float32)  # leftover between callbacks

    def callback(outdata, frames, time_info, status):
        nonlocal carry
        if status:
            print(f"\n[audio status] {status}")
        while carry.shape[0] < frames:
            try:
                carry = np.concatenate([carry, chunks.get_nowait()], axis=0)
            except queue.Empty:
                break
        n = min(frames, carry.shape[0])
        outdata[:n] = carry[:n]
        if n < frames:
            outdata[n:] = 0.0
            print("\n[underrun] generation fell behind")
        carry = carry[n:]

    def queued_samples():
        return sum(c.shape[0] for c in list(chunks.queue))

    # Pre-roll a cushion (by audio seconds) before opening the device.
    while queued_samples() < lead_samples:
        frames, drums = next(plan)
        wav, state = mrt.generate(style=embedding, frames=frames, state=state, drums=drums)
        chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=channels,
        dtype="float32", blocksize=0, callback=callback,
    )
    print(f"Beat-synced streaming @ {sample_rate} Hz, {bpm} BPM {time_sig} (Ctrl+C to stop)...")
    try:
        stream.start()
        # Generate beats on the main thread; keep ~lead_samples queued so the
        # device never starves. Sleep briefly when ahead instead of busy-looping.
        while True:
            if queued_samples() < lead_samples:
                frames, drums = next(plan)
                wav, state = mrt.generate(style=embedding, frames=frames, state=state, drums=drums)
                chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="warm ambient synth pads")
    p.add_argument("--tempo", type=float, default=100.0, help="target tempo in BPM (drives the beat grid + soft prompt hint)")
    p.add_argument("--time-sig", choices=TIME_SIGS, default="4/4", help="time signature (drives the backbeat grid + soft prompt hint)")
    p.add_argument("--size", default="mrt2_base")  # 2.4B: full quality, real-time on M4 Max
    args = p.parse_args()

    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    effective_prompt = prompt_with_tempo(args.prompt, args.tempo, args.time_sig)
    print(f"Embedding prompt: {effective_prompt!r}")
    embedding = mrt.embed_style(effective_prompt)

    stream_beats(mrt, embedding, args.tempo, args.time_sig)


if __name__ == "__main__":
    main()
