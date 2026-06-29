"""tf.data-Pipeline: Track-Pfade + Split → gepaarte Spektrogramme (Streaming).

Zwei Varianten:
* **Train** (:func:`make_dataset`): Shuffle + Augmentation + Zufalls-Crops, ``repeat``.
* **Val/Test** (:func:`make_eval_dataset`): **deterministisch** — feste, gleichmäßig
  verteilte Segmente pro Track, ohne Shuffle/Repeat/Augmentation → reproduzierbare Zahl.

Pro Beispiel: Track wählen → je Stem Crop via ``soundfile`` (Frame-Offset) → **Mono-
Downmix** → (Augmentation) → ``bandlimit`` → STFT beider → ``compress`` → ``drop_nyquist`` →
``copy_up_hf`` (nur Input) → Re/Im stapeln + Freq-Koord-Kanal.
Ausgabe je Beispiel: ``(input_spec [512, T, 3], target_spec [512, T, 2])`` mit ``T = SEG_FRAMES``.

**On-the-fly-Resampling:** Ist die native Samplerate ≠ ``cfg.SR`` (z. B. 44,1-kHz-MUSDB18-HQ
auf Kaggle), wird der Crop in nativen Frames gelesen und mit soxr auf 32 kHz resampelt — derselbe
Code läuft so lokal (32-kHz-Cache = No-op) und auf Kaggle.

Augmentationen (nur Train): Kanalwahl mean/left/right, Stem-Remix/Gain/Polarität (``augment.py``),
variabler Cutoff (``cfg.CUTOFFS_HZ``), wechselnder Crop-Offset.
"""

from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

from bwe import config as cfg
from bwe.data.augment import augment_target
from bwe.data.splits import TrackInfo, get_split
from bwe.dsp.bandlimit import bandlimit
from bwe.dsp.compress import compress
from bwe.dsp.copyup import copy_up_hf
from bwe.dsp.stft import drop_nyquist, stft

DOWNMIX_MODES = ("mean", "left", "right")

# Frequenz-Koordinatenkanal: jedem Bin seine (normierte) Frequenzposition mitgeben.
_FREQ_COORD = tf.cast(tf.linspace(0.0, 1.0, cfg.N_BINS_NET), tf.float32)[:, tf.newaxis]


# --------------------------------------------------------------------------- #
# Audio-Laden (Python, läuft via tf.numpy_function)
# --------------------------------------------------------------------------- #
def _seg_native(sr: int) -> int:
    """Crop-Länge in nativen Frames, die nach Resampling ~SEG_SAMPLES @ cfg.SR ergibt."""
    return cfg.SEG_SAMPLES if sr == cfg.SR else int(np.ceil(cfg.SEG_SAMPLES * sr / cfg.SR))


def _read_crop_mono(path: str, start: int, length: int, mode: str, sr: int) -> np.ndarray:
    """Crop (native Frames) → Mono-Downmix → Resample auf cfg.SR → exakt SEG_SAMPLES."""
    data, _ = sf.read(path, start=start, frames=length, always_2d=True, dtype="float32")
    if mode == "left":
        mono = data[:, 0]
    elif mode == "right":
        mono = data[:, -1]
    else:
        mono = data.mean(axis=1)
    if sr != cfg.SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=cfg.SR, res_type="soxr_hq")
    seg = cfg.SEG_SAMPLES
    mono = mono[:seg] if mono.shape[0] >= seg else np.pad(mono, (0, seg - mono.shape[0]))
    return mono.astype(np.float32)


def _read_target(paths_i, start, length, sr, augment, rng) -> np.ndarray:
    """Alle Stems eines Tracks lesen und zum (ggf. augmentierten) Target mischen."""
    mode = str(rng.choice(DOWNMIX_MODES)) if augment else "mean"
    stems = {s: _read_crop_mono(p, start, length, mode, sr) for s, p in zip(cfg.STEMS, paths_i)}
    return augment_target(stems, rng=rng, enabled=augment)


def _track_paths_info(tracks: list[TrackInfo]):
    paths = [[str(t.stems[s]) for s in cfg.STEMS] for t in tracks]
    infos = [sf.info(p[0]) for p in paths]
    srs = [int(i.samplerate) for i in infos]
    nframes = [int(i.frames) for i in infos]
    return paths, srs, nframes


def _make_random_loader(tracks, augment: bool, cutoff_bins: list[int]):
    """idx -> (target_wave [SEG_SAMPLES], cutoff_bin) mit Zufalls-Crop (Train)."""
    paths, srs, nframes = _track_paths_info(tracks)

    def _load(idx):
        i = int(idx)
        rng = np.random.default_rng()                 # neuer Crop/Mix bei jedem Zugriff
        sr, n = srs[i], nframes[i]
        seg_native = _seg_native(sr)
        start = 0 if n <= seg_native else int(rng.integers(0, n - seg_native + 1))
        length = min(seg_native, n)
        target = _read_target(paths[i], start, length, sr, augment, rng)
        cb = int(rng.choice(cutoff_bins)) if len(cutoff_bins) > 1 else int(cutoff_bins[0])
        return target, np.int32(cb)

    return _load


# --------------------------------------------------------------------------- #
# Wellenform → Modell-Tensoren (TF-Graph)
# --------------------------------------------------------------------------- #
def waveform_to_tensors(target_wave: tf.Tensor, cutoff_bin=cfg.CUTOFF_BIN):
    """Vollband-Target-Wellenform → modellfertige Spektrogramm-**Tensoren**
    ``(input_spec[512, T, 3], target_spec[512, T, 2])``.

    „Tensor" = ``tf.Tensor``: das Array-Format, mit dem TensorFlow rechnet (wie ein
    NumPy-Array, aber im TF-Graphen — GPU-fähig und differenzierbar). Wir arbeiten
    mit TF-Tensoren, weil die ganze Kette (``tf.signal``-STFT, Kompression, Copy-Up)
    und später das U-Net TF-Operationen sind und im Training Gradienten durch sie
    zurückfließen müssen. (Die Roh-Wellenform kommt als NumPy aus ``soundfile`` und
    wird beim Eintritt in den Graphen zu einem Tensor.)

    ``cutoff_bin`` darf ein Skalar-Tensor sein (variabler Cutoff pro Beispiel).
    """
    input_wave = bandlimit(target_wave, cutoff_bin)

    spec_t = drop_nyquist(compress(stft(target_wave)))       # [512, T] complex
    spec_i = drop_nyquist(compress(stft(input_wave)))        # [512, T] complex
    spec_i = copy_up_hf(spec_i, cutoff_bin=cutoff_bin)       # HF per Hochkopie füllen

    target_spec = tf.stack([tf.math.real(spec_t), tf.math.imag(spec_t)], axis=-1)
    ri_in = tf.stack([tf.math.real(spec_i), tf.math.imag(spec_i)], axis=-1)

    t = tf.shape(spec_i)[1]
    freq = tf.tile(_FREQ_COORD[:, tf.newaxis, :], [1, t, 1])   # [512, T, 1]
    input_spec = tf.concat([ri_in, freq], axis=-1)            # [512, T, 3]

    input_spec.set_shape([cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_INPUT_CHANNELS])
    target_spec.set_shape([cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_OUTPUT_CHANNELS])
    return input_spec, target_spec


# --------------------------------------------------------------------------- #
# Öffentliche API — Train
# --------------------------------------------------------------------------- #
def make_dataset(
    split: str,
    batch_size: int = cfg.BATCH_SIZE,
    augment: bool | None = None,
    cutoffs=None,
    shuffle: bool = True,
    repeat: bool = True,
    seed: int = cfg.SEED,
    shuffle_buffer: int | None = None,
    limit: int | None = None,
) -> tf.data.Dataset:
    """Train-Dataset: gepaarte Spektrogramme, Shuffle + Augmentation + Zufalls-Crops.

    ``augment=None`` → an für 'train', aus sonst. ``cutoffs=None`` → ``cfg.CUTOFFS_HZ``;
    mehrere Werte = variabler Cutoff pro Beispiel (sonst/aus → der erste).
    ``limit`` schneidet auf die ersten N Tracks (Subset-First-Training).
    """
    tracks = get_split(split)
    if limit is not None:
        tracks = tracks[:limit]
    if not tracks:
        raise RuntimeError(f"Keine Tracks für Split {split!r} unter {cfg.DATA_ROOT}")
    if augment is None:
        augment = split == "train"
    cutoffs = cfg.CUTOFFS_HZ if cutoffs is None else cutoffs
    cutoff_bins = [cfg.cutoff_bin_for(hz) for hz in cutoffs]
    if not augment:
        cutoff_bins = cutoff_bins[:1]

    loader = _make_random_loader(tracks, augment, cutoff_bins)

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(tracks)))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer or len(tracks), seed=seed,
                        reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    def _map(idx):
        wave, cb = tf.numpy_function(loader, [idx], [tf.float32, tf.int32], name="load_target")
        wave = tf.ensure_shape(wave, [cfg.SEG_SAMPLES])
        return waveform_to_tensors(wave, cb)

    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# --------------------------------------------------------------------------- #
# Öffentliche API — Val/Test (deterministisch)
# --------------------------------------------------------------------------- #
def _eval_segments(srs, nframes, segments_per_track):
    """Feste (track_idx, start_native)-Paare: gleichmäßig über jeden Track verteilt."""
    pairs = []
    for i, (sr, n) in enumerate(zip(srs, nframes)):
        seg_native = _seg_native(sr)
        if n <= seg_native:
            starts = [0]
        else:
            starts = np.linspace(0, n - seg_native, segments_per_track).astype(int).tolist()
        pairs.extend((i, int(s)) for s in starts)
    return pairs


def make_eval_dataset(
    split: str,
    segments_per_track: int = cfg.VAL_SEGMENTS_PER_TRACK,
    batch_size: int = cfg.BATCH_SIZE,
    cutoff_hz: int = cfg.CUTOFF_HZ,
    limit: int | None = None,
) -> tf.data.Dataset:
    """Deterministisches Val/Test-Dataset: feste Segmente, keine Augmentation, fester Cutoff.

    Deckt **alle** Tracks ab (je ``segments_per_track`` gleichmäßig verteilte Segmente,
    kurze Tracks ≥1). Keras validiert pro Epoche über das *gesamte* Dataset → der
    Val-Wert mittelt über alle Tracks, nicht über einen einzelnen Batch.
    """
    tracks = get_split(split)
    if limit is not None:
        tracks = tracks[:limit]
    if not tracks:
        raise RuntimeError(f"Keine Tracks für Split {split!r} unter {cfg.DATA_ROOT}")
    paths, srs, nframes = _track_paths_info(tracks)
    pairs = _eval_segments(srs, nframes, segments_per_track)
    cutoff_bin = cfg.cutoff_bin_for(cutoff_hz)

    def _load(idx, start):
        i, s = int(idx), int(start)
        length = min(_seg_native(srs[i]), nframes[i] - s)
        return _read_target(paths[i], s, length, srs[i], augment=False,
                            rng=np.random.default_rng(0))

    ds = tf.data.Dataset.from_tensor_slices(np.asarray(pairs, dtype=np.int64))

    def _map(pair):
        wave = tf.numpy_function(_load, [pair[0], pair[1]], tf.float32, name="eval_load")
        wave = tf.ensure_shape(wave, [cfg.SEG_SAMPLES])
        return waveform_to_tensors(wave, cutoff_bin)

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


# --------------------------------------------------------------------------- #
# Hilfen
# --------------------------------------------------------------------------- #
def steps_per_epoch_for(split: str = "train", batch_size: int = cfg.BATCH_SIZE,
                        limit: int | None = None) -> int:
    """≈ Gesamtdauer_train / (B · Segmentlänge) — im Mittel jede Sekunde ~1×/Epoche."""
    tracks = get_split(split)
    if limit is not None:
        tracks = tracks[:limit]
    total = 0.0
    for t in tracks:
        info = sf.info(str(t.stems[cfg.STEMS[0]]))
        total += info.frames * cfg.SR / info.samplerate        # in 32-k-Samples
    return max(1, int(total / (batch_size * cfg.SEG_SAMPLES)))


def sample_target(split: str, track_index: int = 0, augment: bool = False) -> np.ndarray:
    """Ein einzelnes Target-Wellenform-Beispiel (für Notebook/Tests)."""
    tracks = get_split(split)
    loader = _make_random_loader(tracks, augment, [cfg.CUTOFF_BIN])
    wave, _ = loader(track_index)
    return wave
