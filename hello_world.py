"""Minimal Magenta RealTime 2 hello-world: text prompt -> audio file.

Uses MagentaRT2SystemMlxfn so it loads the already-downloaded .mlxfn weights
(from ~/Documents/Magenta/magenta-rt-v2) with zero network access.\

Run it / listen:
uv run python hello_world.py --prompt "disco funk" --seconds 4 --out out.wav
afplay out.wav

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


def prompt_with_tempo(prompt, bpm):
    """MRT2 has no numeric tempo input; the only lever is the text style prompt.
    Inject both a tempo word and the BPM number as a soft hint to MusicCoCa."""
    n = int(bpm) if float(bpm).is_integer() else bpm  # "100 BPM" not "100.0 BPM"
    return f"{prompt}, {tempo_word(bpm)}, {n} BPM"


def stream_forever(mrt, embedding, chunk_frames=25, lead_chunks=3):
    """Generate ~1s chunks back-to-back, threading state forward, and stream them
    to the speakers gaplessly until Ctrl+C.

    Generation must stay on this (main) thread — the imported .mlxfn function is
    bound to the thread that imported the model, so calling generate() elsewhere
    raises "no Stream(gpu, N) in current thread". So generation runs here and
    fills a small queue; PortAudio's own audio thread runs the callback, which
    only does buffer copies (no MLX). The multi-chunk lead keeps the device from
    starving between generate() calls — the serialized gen→write loop was what
    caused the audible per-chunk seams.

    Generation is ~0.6s per 1s of audio, so the producer stays ahead of playback.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    # Prime first chunk to learn sample rate + channel count.
    wav, state = mrt.generate(style=embedding, frames=chunk_frames)
    sample_rate, channels = wav.sample_rate, wav.num_channels

    chunks: queue.Queue = queue.Queue(maxsize=lead_chunks)  # bounded => backpressure
    chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))

    carry = np.empty((0, channels), dtype=np.float32)  # leftover between callbacks

    def callback(outdata, frames, time_info, status):
        nonlocal carry
        if status:
            print(f"\n[audio status] {status}")
        # Pull whole chunks until we have enough samples for this callback.
        while carry.shape[0] < frames:
            try:
                carry = np.concatenate([carry, chunks.get_nowait()], axis=0)
            except queue.Empty:
                break
        n = min(frames, carry.shape[0])
        outdata[:n] = carry[:n]
        if n < frames:
            outdata[n:] = 0.0  # underrun: pad with silence, never block the callback
            print("\n[underrun] generation fell behind")
        carry = carry[n:]

    # Pre-roll a cushion on this thread before opening the device.
    while chunks.qsize() < lead_chunks:
        wav, state = mrt.generate(style=embedding, frames=chunk_frames, state=state)
        chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=channels,
        dtype="float32", blocksize=0, callback=callback,
    )
    print(f"Streaming @ {sample_rate} Hz (Ctrl+C to stop)...")
    try:
        stream.start()
        # Keep generating on the main thread; the bounded queue's blocking put
        # paces us to playback rate (backpressure), and the callback drains it.
        while True:
            wav, state = mrt.generate(style=embedding, frames=chunk_frames, state=state)
            chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="warm ambient synth pads")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--size", default="mrt2_base")  # 2.4B: full quality, real-time on M4 Max
    p.add_argument("--out", default="out.wav")
    p.add_argument("--tempo", type=float, default=100.0, help="target tempo in BPM (soft hint injected into the prompt; MRT2 has no hard tempo control)")
    p.add_argument("--play", action="store_true", help="play to speakers instead of writing a file (Ctrl+C to stop)")
    p.add_argument("--stream", action="store_true", help="generate + play continuously until Ctrl+C")
    args = p.parse_args()

    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    effective_prompt = prompt_with_tempo(args.prompt, args.tempo)
    print(f"Embedding prompt: {effective_prompt!r}")
    embedding = mrt.embed_style(effective_prompt)

    if args.stream:
        stream_forever(mrt, embedding)
        return

    frames = round(args.seconds * 25)  # 25 frames == 1 second
    print(f"Generating ~{args.seconds}s ({frames} frames)...")
    wav, _state = mrt.generate(style=embedding, frames=frames)

    if args.play:
        import sounddevice as sd
        print(f"Playing @ {wav.sample_rate} Hz (Ctrl+C to stop)...")
        try:
            sd.play(wav.samples, wav.sample_rate)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("\nStopped.")
    else:
        wav.write(args.out)
        print(f"Wrote {args.out} @ {wav.sample_rate} Hz")


if __name__ == "__main__":
    main()
