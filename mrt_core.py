"""Magenta RealTime 2 core library: prompt/tempo hints, key detection, the beat
grid, device resolution, live input capture, and the MRT2 system loader.

Loaded via MagentaRT2SystemMlxfn so it uses the already-downloaded .mlxfn weights
(from ~/Documents/Magenta/magenta-rt-v2) with zero network access.

This is a library only — no CLI. The active entry point is staged_band.py, which
imports the building blocks defined here. (probe.py / probe_rtf.py also import
from this module.)
"""
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
