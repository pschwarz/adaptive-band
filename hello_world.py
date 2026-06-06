"""Magenta RealTime 2 beat-synced live stream: text prompt -> endless audio.

Uses MagentaRT2SystemMlxfn so it loads the already-downloaded .mlxfn weights
(from ~/Documents/Magenta/magenta-rt-v2) with zero network access.

Run it (Ctrl+C to stop):
uv run python hello_world.py --prompt "blues" --tempo 100 --time-sig 4/4
uv run python hello_world.py --prompt "jazz trio" --listen        # follow live input's key
uv run python hello_world.py --prompt "pads" --key C --mode dorian # static scale lock
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


# --- Key / scale / mode ---------------------------------------------------
# MRT2 has no symbolic key input. The pitch lever is generate(notes=<128 ints>),
# a per-MIDI-pitch state map (-1 masked / 0 off / 1 on / 2 onset / 3 on, model's
# choice). A key is a rule over pitch classes; we expand (root, mode) into that
# 128-slot vector: in-scale pitches = 3 (model's choice), out-of-scale = -1
# (masked / "no opinion") — a SOFT scale bias. Using 0 (off) here is a *hard*
# lock that forbids every out-of-scale pitch on every beat; applied continuously
# with the streaming state carried forward, that contradicts the model's own
# evolving line and it loses coherence within ~10s. -1 leaves the model free to
# pass through chromatic/leading tones while the prompt + cfg_notes still bias it
# toward the scale.

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODE_INTERVALS = {  # semitone offsets from the mode's root
    "ionian":     (0, 2, 4, 5, 7, 9, 11),  # = major
    "dorian":     (0, 2, 3, 5, 7, 9, 10),
    "phrygian":   (0, 1, 3, 5, 7, 8, 10),
    "lydian":     (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian":    (0, 2, 3, 5, 7, 8, 10),  # = natural minor
    "locrian":    (0, 1, 3, 5, 6, 8, 10),
}
MODE_ALIASES = {"major": "ionian", "minor": "aeolian"}
MODES = tuple(MODE_INTERVALS)
# Friendlier names for display only (the common modes have well-known names).
MODE_DISPLAY = {"ionian": "major", "aeolian": "minor"}


def mode_name(mode):
    """Display name for a mode: ionian->major, aeolian->minor, else unchanged."""
    return MODE_DISPLAY.get(mode, mode)


def note_index(name):
    """'C', 'F#', 'Bb' -> pitch class 0..11."""
    name = name.strip().capitalize().replace("Db", "C#").replace("Eb", "D#") \
        .replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")
    return NOTE_NAMES.index(name)


def scale_mask(root, mode):
    """(root pitch class, mode name) -> 128-int notes vector for generate().
    In-scale pitches = 3 (model chooses onset/continuation), out-of-scale = -1
    (masked / no opinion) — a SOFT scale bias, not a hard lock (see the encoding
    note above: 0/"off" forbids out-of-scale pitches every beat and the model
    loses coherence within ~10s under the streaming state)."""
    mode = MODE_ALIASES.get(mode, mode)
    allowed = {(root + iv) % 12 for iv in MODE_INTERVALS[mode]}
    return [3 if (p % 12) in allowed else -1 for p in range(128)]


# Per-scale-degree weights for the key templates (index = semitones above the
# mode root, only scale degrees used). Tonic and fifth are heaviest, the
# characteristic/colour tones next — this is what lets us tell a mode from its
# relative modes (which share the same pitch-class SET; a flat 0/1 template
# cannot, see estimate_key). A pragmatic Krumhansl-style salience profile.
_DEGREE_WEIGHT = {0: 5.0, 7: 3.5}  # tonic, perfect fifth
_DEFAULT_DEGREE_WEIGHT = 2.0       # any other in-scale degree

# The two "common" modes. They get a score bonus so the estimator defaults to
# major/minor and only reports an exotic mode when the chroma evidence is clearly
# stronger — see MODAL_MARGIN. (ionian = major, aeolian = natural minor.)
COMMON_MODES = ("ionian", "aeolian")
# How much extra correlation a non-major/minor mode must beat the best
# major/minor candidate by, expressed as a fraction of the chroma's own scale
# (||c||), to be reported. 0 = no bias (old behavior); higher = stickier to
# major/minor. Tuned by ear; exposed as --modal-margin.
MODAL_MARGIN = 0.6


def _key_template(root, mode):
    import numpy as np

    t = np.zeros(12)
    for iv in MODE_INTERVALS[mode]:
        t[(root + iv) % 12] = _DEGREE_WEIGHT.get(iv, _DEFAULT_DEGREE_WEIGHT)
    return t - t.mean()


def estimate_key(chroma_mean, modal_margin=MODAL_MARGIN, return_scores=False):
    """12-vector mean chroma -> (root, mode). Modal generalization of
    Krumhansl-Schmuckler: correlate the (zero-meaned) chroma against 84 weighted
    diatonic templates (12 roots x 7 modes), pick the best fit. Templates weight
    tonic+fifth heavily so a mode is distinguishable from its relative modes
    (which share the same pitch-class set).

    Because the exotic modes are fragile (dorian/aeolian differ by one pitch
    class, the 6th), we bias toward the common modes: an exotic mode is only
    reported if its score beats the best major/minor candidate by at least
    `modal_margin * ||c||`. Otherwise the nearest major/minor wins. Set
    modal_margin=0 for the raw best-fit.

    With return_scores=True, returns (winner, scores, margin, cnorm) where scores
    is a {(root, mode): score} dict, margin is the absolute bonus an exotic mode
    had to clear, and cnorm is ||c|| (the chroma's scale) — so the gap between
    candidates can be re-expressed in --modal-margin units. For debugging."""
    import numpy as np

    c = np.asarray(chroma_mean, dtype=np.float64)
    c = c - c.mean()
    cnorm = float(np.linalg.norm(c)) or 1.0
    margin = modal_margin * cnorm

    scores = {}
    for root in range(12):
        for mode in MODES:
            scores[(root, mode)] = float((c * _key_template(root, mode)).sum())

    best_common = max(
        ((k, s) for k, s in scores.items() if k[1] in COMMON_MODES),
        key=lambda kv: kv[1],
    )
    best_any = max(scores.items(), key=lambda kv: kv[1])

    # Take the exotic winner only if it clears major/minor by the margin.
    if best_any[0][1] not in COMMON_MODES and best_any[1] >= best_common[1] + margin:
        winner = best_any[0]
    else:
        winner = best_common[0]
    if return_scores:
        return winner, scores, margin, cnorm
    return winner


def format_scores(scores, winner, margin, cnorm, top=8):
    """One-line ranked dump of the top-N (root, mode) candidates for debugging.
    Marks the winner (*) and shows two margins: the absolute bonus an exotic mode
    had to clear, and `gap` — the best-exotic-minus-best-common score difference
    expressed as a fraction of ||c||. gap is in the SAME units as --modal-margin:
    setting --modal-margin just above the gap you see keeps it on major/minor;
    below it lets that exotic mode through. (gap<0 means the exotic mode already
    lost outright — common mode also outscores it on raw points.)"""
    best_common = max((s for k, s in scores.items() if k[1] in COMMON_MODES))
    best_exotic = max((s for k, s in scores.items() if k[1] not in COMMON_MODES))
    gap = (best_exotic - best_common) / cnorm
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]
    parts = []
    for (root, mode), s in ranked:
        mark = "*" if (root, mode) == winner else " "
        parts.append(f"{mark}{NOTE_NAMES[root]} {mode_name(mode)} {s:.2f}")
    return f"(margin {margin:.2f}, gap {gap:+.2f} of ||c||)  " + " | ".join(parts)


# --- basic-pitch front-end (alternative to librosa chroma) ----------------
# Spotify's basic-pitch polyphonic note model, run via its bundled ONNX file so
# we avoid the pip package (which drags in TensorFlow + pins numpy<2 and breaks
# MRT2/MLX, which needs numpy 2). Only onnxruntime is needed. The model outputs a
# note posteriorgram; we threshold it and fold active pitches into a 12-vector
# pitch-class histogram that feeds the same estimate_key as chroma does.
import os

BASIC_PITCH_MODEL = os.path.join(os.path.dirname(__file__), "models", "nmp.onnx")
BP_SR = 22050        # the model's native sample rate
BP_SAMPLES = 43844   # fixed ~1.99s input window
NOTE_THRESHOLD = 0.3  # min note-posterior to count a pitch (tune by ear)
_bp_session = None    # lazily-loaded ONNX session (load cost only when used)


def _basic_pitch_session():
    global _bp_session
    if _bp_session is None:
        import onnxruntime as ort
        _bp_session = ort.InferenceSession(
            BASIC_PITCH_MODEL, providers=["CPUExecutionProvider"])
    return _bp_session


def basic_pitch_histogram(y, sr, note_threshold=NOTE_THRESHOLD):
    """Live audio (mono, any sr) -> 12-vector pitch-class histogram via the
    basic-pitch ONNX note model. Resamples to 22050 Hz, then runs the model over
    consecutive ~2s chunks spanning the WHOLE window (the model input is a fixed
    ~2s tensor), thresholds each note posteriorgram, and sums confident frames per
    pitch folded into 12 classes. Tiling matters for single-note playing (e.g.
    bass outlining a chord over several seconds): a longer window accumulates the
    notes into one clean key picture. Same output shape/role as the chroma mean."""
    import numpy as np
    import librosa

    if sr != BP_SR:
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=BP_SR)
    y = np.asarray(y, dtype=np.float32)
    if y.shape[0] < BP_SAMPLES:               # left-pad short windows
        y = np.concatenate([np.zeros(BP_SAMPLES - y.shape[0], np.float32), y])

    # Tile into back-to-back ~2s chunks covering the full window (last chunk
    # right-aligned so the most recent audio is always included).
    s = _basic_pitch_session()
    name = s.get_inputs()[0].name
    n = y.shape[0]
    starts = list(range(0, max(1, n - BP_SAMPLES + 1), BP_SAMPLES))
    if starts[-1] + BP_SAMPLES < n:
        starts.append(n - BP_SAMPLES)
    pc = np.zeros(12)
    for st in starts:
        x = y[st:st + BP_SAMPLES].reshape(1, BP_SAMPLES, 1)
        note = s.run(None, {name: x})[0][0]   # (frames, 88)
        act = (note >= note_threshold).sum(axis=0).astype(np.float64)
        for p in range(act.shape[0]):         # pitch row p -> MIDI 21+p (A0=21)
            pc[(21 + p) % 12] += act[p]
    return pc


# beats-per-bar and the (1-based) beats that get the backbeat drum tag.
BEAT_GRID = {
    "4/4": (4, {2, 4}),
    "3/4": (3, {2, 3}),
    "6/8": (2, {2}),  # compound: 2 dotted-quarter beats, backbeat on beat 2
}


def beat_plan(bpm, time_sig):
    """Yield (frames, drums, bar_start) per generate() call, one beat at a time,
    forever. bar_start is True on the first slice of beat 1 — the bar boundary
    where the live key/mask is re-derived.

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
        bar_start = beat == 1
        acc += spb
        n = round(acc)
        acc -= n
        if n < 1:
            n = 1
        if beat in backbeats and n >= 2:
            yield (1, [1], bar_start)     # onset
            yield (n - 1, None, False)    # tail
        else:
            yield (n, [1] if beat in backbeats else None, bar_start)


def resolve_input_device(spec):
    """spec is an int index or a substring of a device name. Returns the index
    of a matching input-capable device (raises if none)."""
    import sounddevice as sd

    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    spec_l = str(spec).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec_l in d["name"].lower():
            return i
    raise ValueError(f"no input device matching {spec!r}")


def resolve_output_device(spec):
    """spec is an int index or a substring of a device name. Returns the index
    of a matching output-capable device (raises if none)."""
    import sounddevice as sd

    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    spec_l = str(spec).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and spec_l in d["name"].lower():
            return i
    raise ValueError(f"no output device matching {spec!r}")


class InputCapture:
    """Live audio in -> rolling chroma. A second PortAudio InputStream fills a
    fixed-size ring buffer (callback only memcpys — no MLX, same thread discipline
    as the output callback). current_key() snapshots the buffer on the main thread
    and returns (root, mode), or None when the input is near-silent."""

    def __init__(self, device, window_seconds, channel=0, rms_gate=1e-3,
                 modal_margin=MODAL_MARGIN, estimator="chroma",
                 note_threshold=NOTE_THRESHOLD, debug_scores=False):
        import numpy as np
        import sounddevice as sd

        self._np = np
        self._channel = channel
        self._modal_margin = modal_margin
        self._estimator = estimator
        self._note_threshold = note_threshold
        self._debug_scores = debug_scores
        self._window_seconds = window_seconds
        idx = resolve_input_device(device)
        info = sd.query_devices(idx)
        self.idx = idx
        self.in_channels = int(info["max_input_channels"])
        self.sr = int(info["default_samplerate"])
        self.name = info["name"]
        self.rms_gate = rms_gate
        self._alloc_buffer()
        # Our own InputStream, separate from (and independently clocked from) the
        # output stream. They must be different devices — two streams on one
        # CoreAudio device fail with -10863, and a shared duplex stream slaves the
        # output to the input clock, which drifts the beat grid over time.
        # Open all device input channels; we pick `channel` in the callback (an
        # iD4 has 4 inputs and the live signal may not be on channel 0).
        self._stream = sd.InputStream(
            device=idx, channels=self.in_channels, samplerate=self.sr,
            dtype="float32", blocksize=0, callback=self._callback,
        )

    def _alloc_buffer(self):
        n = int(self._window_seconds * self.sr)
        self._buf = self._np.zeros(n, dtype=self._np.float32)  # mono ring buffer
        self._pos = 0

    def _callback(self, indata, frames, time_info, status):
        self.feed(indata, frames)

    def feed(self, indata, frames):
        """Write one input block into the ring buffer. Copy-only — no MLX, same
        thread discipline as the output path."""
        np = self._np
        x = indata[:, self._channel]
        n = self._buf.shape[0]
        pos = self._pos
        if frames >= n:           # incoming bigger than buffer: keep the tail
            self._buf[:] = x[-n:]
            self._pos = 0
            return
        end = pos + frames
        if end <= n:
            self._buf[pos:end] = x
        else:                     # wrap
            k = n - pos
            self._buf[pos:] = x[:k]
            self._buf[:end - n] = x[k:]
        self._pos = end % n

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def current_key(self):
        np = self._np
        snap = self._buf.copy()  # atomic enough: one snapshot, slight tear is harmless
        if float(np.sqrt(np.mean(snap ** 2))) < self.rms_gate:
            return None          # silent: caller keeps the previous mask
        if self._estimator == "basic-pitch":
            hist = basic_pitch_histogram(snap, self.sr, self._note_threshold)
        else:
            import librosa
            hist = librosa.feature.chroma_cqt(y=snap, sr=self.sr).mean(axis=1)
        if self._debug_scores:
            winner, scores, margin, cnorm = estimate_key(
                hist, modal_margin=self._modal_margin, return_scores=True)
            print("  " + format_scores(scores, winner, margin, cnorm))
            return winner
        return estimate_key(hist, modal_margin=self._modal_margin)


def monitor_input(device, channel=0, seconds=None, estimator="chroma",
                  note_threshold=NOTE_THRESHOLD, window_seconds=2.0,
                  debug_scores=False):
    """Diagnostic: pass the chosen input channel straight to the speakers and
    print a live RMS meter + the key estimate. Use this to confirm audio is
    actually arriving (and at what level) before debugging key detection. Honors
    --estimator / --note-threshold / --listen-seconds so it matches the live path.
    Runs until Ctrl+C (or `seconds` if given). No model needed."""
    import numpy as np
    import sounddevice as sd

    idx = resolve_input_device(device)
    info = sd.query_devices(idx)
    sr = int(info["default_samplerate"])
    in_ch = int(info["max_input_channels"])
    print(f"Monitoring {info['name']!r} @ {sr} Hz, {in_ch} in-channels, "
          f"listening on channel {channel} ({estimator} estimator, "
          f"{window_seconds:g}s window; Ctrl+C to stop)...")
    print("Play something — you should HEAR it and see the level move.")
    print("If a channel stays flat while you play, try --input-channel 1/2/3.\n")

    # Rolling buffer for the key estimate + peak tracking.
    win = np.zeros(int(window_seconds * sr), dtype=np.float32)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"\n[status] {status}")
        x = indata[:, channel]
        outdata[:, 0] = x                       # passthrough to speakers
        if outdata.shape[1] > 1:
            outdata[:, 1] = x
        nonlocal win
        win = np.roll(win, -frames)
        win[-frames:] = x
        # Per-channel peaks so you can see WHICH input is hot.
        peaks = np.max(np.abs(indata), axis=0) if frames else np.zeros(indata.shape[1])
        rms = float(np.sqrt(np.mean(x ** 2)))
        bars = int(min(rms * 400, 40))
        pk = " ".join(f"{p:.2f}" for p in peaks)
        print(f"\rRMS {rms:7.4f} |{'#' * bars:<40}| ch-peaks[{pk}]", end="", flush=True)

    stream = sd.Stream(
        device=(idx, None), channels=(in_ch, None), samplerate=sr,
        dtype="float32", blocksize=0, callback=callback,
    )
    try:
        stream.start()
        import time as _t
        t0 = _t.time()
        while seconds is None or _t.time() - t0 < seconds:
            _t.sleep(0.5)
            snap = win.copy()
            if float(np.sqrt(np.mean(snap ** 2))) >= 1e-3:
                if estimator == "basic-pitch":
                    hist = basic_pitch_histogram(snap, sr, note_threshold)
                else:
                    import librosa
                    hist = librosa.feature.chroma_cqt(y=snap, sr=sr).mean(axis=1)
                if debug_scores:
                    (r, m), scores, margin, cnorm = estimate_key(hist, return_scores=True)
                    print(f"\nkey~ {NOTE_NAMES[r]} {mode_name(m)}   "
                          + format_scores(scores, (r, m), margin, cnorm))
                else:
                    r, m = estimate_key(hist)
                    print(f"   key~ {NOTE_NAMES[r]} {mode_name(m)}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


def stream_beats(mrt, embedding, bpm, time_sig, lead_seconds=1.5,
                 capture=None, cfg_notes=None, init_mask=None, output_device=None,
                 cfg_musiccoca=None, temperature=None, top_k=None, reset_seconds=0.0):
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

    If `capture` is given, the scale mask is re-derived from live input at each bar
    boundary (see beat_plan's bar_start) and passed as notes= to every generate().
    Otherwise the static `init_mask` (possibly None) is used throughout.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    plan = beat_plan(bpm, time_sig)
    mask = init_mask
    cur_key = None  # last detected (root, mode); print only when it changes

    def relock(bar_start):
        """At a bar boundary with a live capture, re-derive the scale mask and
        announce the key only when it actually changes."""
        nonlocal mask, cur_key
        if not (bar_start and capture is not None):
            return
        key = capture.current_key()
        if key is not None and key != cur_key:
            cur_key = key
            root, mode = key
            mask = scale_mask(root, mode)
            print(f"\n[key] {NOTE_NAMES[root]} {mode_name(mode)}")

    def gen(frames, drums, state=None):
        return mrt.generate(style=embedding, frames=frames, state=state,
                            drums=drums, notes=mask, cfg_notes=cfg_notes,
                            cfg_musiccoca=cfg_musiccoca, temperature=temperature,
                            top_k=top_k)

    # MRT conditions each step on a rolling context of its own recent output; run
    # open-loop long enough and that self-conditioning drifts (coherence decays
    # after ~10s). Optionally drop the streaming state at a bar boundary every
    # reset_seconds, re-grounding generation on the style embedding. Reset is
    # bar-aligned to hide the seam; tracked in generated-audio seconds (25 fps).
    FPS = 25.0
    gen_seconds = [0.0]      # total generated-audio seconds (mutable for closure)
    last_reset = [0.0]       # gen_seconds at the last state reset

    def maybe_reset(state, frames, bar_start):
        gen_seconds[0] += frames / FPS
        if (reset_seconds and bar_start
                and gen_seconds[0] - last_reset[0] >= reset_seconds):
            last_reset[0] = gen_seconds[0]
            print(f"\n[reset] re-grounding on style @ {gen_seconds[0]:.1f}s")
            return None      # fresh context next gen()
        return state

    # The capture runs its own InputStream (independent clock). It must be a
    # different device from the output stream below: same-device collides
    # (-10863), and a shared duplex stream would slave the output to the input
    # clock and drift the beat grid. main() guards device != output_device.
    if capture is not None:
        capture.start()

    # Prime first slice to learn sample rate + channel count.
    frames, drums, bar_start = next(plan)
    relock(bar_start)
    wav, state = gen(frames, drums)
    gen_seconds[0] += frames / FPS  # account the primed slice in the reset clock
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
        frames, drums, bar_start = next(plan)
        relock(bar_start)
        state = maybe_reset(state, frames, bar_start)
        wav, state = gen(frames, drums, state)
        chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))

    out_idx = resolve_output_device(output_device) if output_device else None
    stream = sd.OutputStream(
        device=out_idx, samplerate=sample_rate, channels=channels,
        dtype="float32", blocksize=0, callback=callback,
    )
    print(f"Beat-synced streaming @ {sample_rate} Hz, {bpm} BPM {time_sig} (Ctrl+C to stop)...")
    try:
        stream.start()
        # Generate beats on the main thread; keep ~lead_samples queued so the
        # device never starves. Sleep briefly when ahead instead of busy-looping.
        while True:
            if queued_samples() < lead_samples:
                frames, drums, bar_start = next(plan)
                relock(bar_start)
                state = maybe_reset(state, frames, bar_start)
                wav, state = gen(frames, drums, state)
                chunks.put(np.ascontiguousarray(wav.samples, dtype=np.float32))
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()
        if capture is not None:
            capture.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="warm ambient synth pads")
    p.add_argument("--tempo", type=float, default=100.0, help="target tempo in BPM (drives the beat grid + soft prompt hint)")
    p.add_argument("--time-sig", choices=TIME_SIGS, default="4/4", help="time signature (drives the backbeat grid + soft prompt hint)")
    p.add_argument("--size", default="mrt2_base")  # 2.4B: full quality, real-time on M4 Max
    # Key / scale / mode (constrains MRT2's pitch output to a scale via notes=).
    p.add_argument("--listen", action="store_true",
                   help="derive key/mode from live audio in and follow it (per bar)")
    p.add_argument("--input-device", default="iD4",
                   help="input device for --listen: index or name substring")
    p.add_argument("--output-device", default=None,
                   help="output device: index or name substring. With --listen, set this "
                        "to a device OTHER than --input-device (the input and output streams "
                        "must not share a device — same-device collides, -10863). Default: "
                        "system default output.")
    p.add_argument("--analysis-bars", type=float, default=4.0,
                   help="how many bars of audio to analyze for key (wider = stabler mode)")
    p.add_argument("--listen-seconds", type=float, default=0.0,
                   help="override the analysis window length in seconds (0 = derive from "
                        "--analysis-bars). Longer helps single-note/bass playing accumulate a key.")
    p.add_argument("--key-strength", type=float, default=0.0,
                   help="cfg_notes scale-bias strength [-1.0..7.0]; 0 = soft bias from the "
                        "mask alone (the model stays coherent), higher pushes harder toward "
                        "the scale but a strong continuous push degrades coherence over time")
    # Long-run coherence: MRT conditions on a rolling context of its own output and
    # drifts open-loop after ~10s. These re-ground / shape generation over time.
    p.add_argument("--reset-seconds", type=float, default=0.0,
                   help="re-ground generation on the style prompt every N seconds (drop the "
                        "streaming state at a bar boundary). 0 = never. Try ~8 if output loses "
                        "coherence over time; cost is a brief seam at each reset.")
    p.add_argument("--style-strength", type=float, default=None,
                   help="cfg_musiccoca [-1.0..7.0] (model default 3.0): how strongly the style "
                        "prompt re-grounds each step. Higher resists open-loop drift but can get "
                        "rigid/artifacty.")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (model default 1.3); lower = steadier/less wandering")
    p.add_argument("--top-k", type=int, default=None,
                   help="top-k sampling (model default 40); lower = more conservative")
    p.add_argument("--key", help="static key root, e.g. C, F#, Bb (skips --listen)")
    p.add_argument("--mode", default="ionian",
                   help=f"static mode for --key: {', '.join(MODES)} (or major/minor)")
    p.add_argument("--input-channel", type=int, default=0,
                   help="which input channel to listen on (iD4 has 4; try 0/1/2/3)")
    p.add_argument("--rms-gate", type=float, default=1e-3,
                   help="min RMS to attempt key detection (lower if signal is quiet)")
    p.add_argument("--modal-margin", type=float, default=MODAL_MARGIN,
                   help="how strongly to favor major/minor; an exotic mode needs to beat "
                        "the best major/minor by this much (0 = no bias, higher = stickier)")
    p.add_argument("--estimator", choices=["chroma", "basic-pitch"], default="chroma",
                   help="key front-end: chroma (librosa, fast) or basic-pitch (ONNX note model, sharper)")
    p.add_argument("--note-threshold", type=float, default=NOTE_THRESHOLD,
                   help="basic-pitch only: min note-posterior to count a pitch (tune by ear)")
    p.add_argument("--debug-scores", action="store_true",
                   help="print the ranked (root, mode) scores each estimate, to see how close "
                        "the candidates are (works with --monitor and --listen)")
    p.add_argument("--monitor", action="store_true",
                   help="diagnostic: passthrough input to speakers + per-channel meter + key estimate (no model)")
    args = p.parse_args()

    if args.monitor:  # diagnostic short-circuit: no model load
        monitor_input(args.input_device, channel=args.input_channel,
                      estimator=args.estimator, note_threshold=args.note_threshold,
                      window_seconds=args.listen_seconds or 2.0,
                      debug_scores=args.debug_scores)
        return

    print(f"Loading {args.size} from exported .mlxfn (no download)...")
    mrt = MagentaRT2SystemMlxfn(size=args.size)

    # Resolve the pitch-conditioning source: static --key wins; else --listen; else none.
    capture, init_mask, key_words = None, None, None
    if args.key:
        root, mode = note_index(args.key), MODE_ALIASES.get(args.mode, args.mode)
        init_mask = scale_mask(root, mode)
        key_words = f"{NOTE_NAMES[root]} {mode_name(mode)}"
        print(f"Static key: {key_words}")
    elif args.listen:
        seconds_per_bar = (60.0 / args.tempo) * BEAT_GRID[args.time_sig][0]
        window = args.listen_seconds or (args.analysis_bars * seconds_per_bar)
        capture = InputCapture(args.input_device, window,
                               channel=args.input_channel, rms_gate=args.rms_gate,
                               modal_margin=args.modal_margin, estimator=args.estimator,
                               note_threshold=args.note_threshold,
                               debug_scores=args.debug_scores)
        # Input and output must be different devices: two streams on one CoreAudio
        # device fail with -10863. Resolve the *effective* output (explicit flag,
        # else the system default) and check it isn't the listen device.
        import sounddevice as sd
        if args.output_device:
            out_idx = resolve_output_device(args.output_device)
        else:
            out_idx = sd.default.device[1]  # system default output index
        if out_idx == capture.idx:
            raise SystemExit(
                f"--listen output device {sd.query_devices(out_idx)['name']!r} is the "
                f"same as the listen input ({capture.name!r}); two PortAudio streams on "
                f"one CoreAudio device collide (-10863), and a shared duplex stream would "
                f"slave output to the input clock and drift the beat. Route output to a "
                f"DIFFERENT device: switch the macOS system output (e.g. to 'External "
                f"Headphones'), or pass --output-device <name|idx>.")
        args.output_device = out_idx  # pin it so OutputStream uses this exact device
        print(f"Listening on {capture.name!r} @ {capture.sr} Hz, channel {args.input_channel} "
              f"({args.analysis_bars:g}-bar window, {args.estimator} estimator); "
              f"key follows live input.")

    effective_prompt = prompt_with_tempo(args.prompt, args.tempo, args.time_sig)
    if key_words:  # reinforce a static key in the text prompt (live mode: mask only)
        effective_prompt += f", in {key_words}"
    print(f"Embedding prompt: {effective_prompt!r}")
    embedding = mrt.embed_style(effective_prompt)

    cfg_notes = args.key_strength if (capture is not None or init_mask is not None) else None
    stream_beats(mrt, embedding, args.tempo, args.time_sig,
                 capture=capture, cfg_notes=cfg_notes, init_mask=init_mask,
                 output_device=args.output_device,
                 cfg_musiccoca=args.style_strength, temperature=args.temperature,
                 top_k=args.top_k, reset_seconds=args.reset_seconds)


if __name__ == "__main__":
    main()
