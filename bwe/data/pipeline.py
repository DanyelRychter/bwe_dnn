"""tf.data-Pipeline: Track-Pfade + Split → gepaarte Spektrogramme (Streaming).

Pro Schritt (lazy, überlappt mit dem Training):
  Track wählen → je Stem **Zufalls-Crop** an zufälligem Frame-Offset via
  ``soundfile`` (nicht den ganzen Track laden) → **Mono-Downmix** → Augmentation
  auf dem Target → ``bandlimit`` (= Input) → STFT beider → ``compress`` →
  ``drop_nyquist`` → ``copy_up_hf`` (nur Input) → Re/Im stapeln + Freq-Koord-Kanal.

Ausgabe je Beispiel: ``(input_spec [512, T, 3], target_spec [512, T, 2])`` mit
``T = SEG_FRAMES``.

Augmentationen (nur bei ``augment=True``):
* **Kanalwahl** ``mean`` / ``left`` / ``right`` — statt nur ``(L+R)/2`` werden auch
  L und R einzeln als Mono-Signal genutzt (≈ verdreifacht die „Sichten" je Crop).
  Für ein Beispiel gilt EINE Wahl für alle Stems (sonst inkohärenter Mix).
* **Variabler Cutoff** — pro Beispiel ein zufälliger Cutoff aus ``cfg.CUTOFFS_HZ``.
  Bei nur einem Eintrag (Default 8 kHz) ändert sich nichts; weitere Werte dort
  eintragen genügt. (Echtes Multi-Cutoff-Training: zusätzlich Masken-Kanal, v2.)
* **Crop-Offset / Stem-Remix / Gain / Polarität** — siehe ``augment.py``.
"""

from __future__ import annotations

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
def _read_crop_mono(path: str, start: int, length: int, mode: str = "mean") -> np.ndarray:
    """Liest nur ``[start:start+length]`` und reduziert zu Mono ``[length]``.

    ``mode``: ``mean`` = (L+R)/2, ``left`` = Kanal 0, ``right`` = letzter Kanal.
    """
    data, _ = sf.read(path, start=start, frames=length, always_2d=True, dtype="float32")
    if mode == "left":
        mono = data[:, 0]
    elif mode == "right":
        mono = data[:, -1]
    else:  # mean
        mono = data.mean(axis=1)
    if mono.shape[0] < cfg.SEG_SAMPLES:
        mono = np.pad(mono, (0, cfg.SEG_SAMPLES - mono.shape[0]))
    return mono.astype(np.float32)


def _make_loader(tracks: list[TrackInfo], augment: bool, cutoff_bins: list[int]):
    """Erzeugt eine Python-Funktion ``idx -> (target_wave [SEG_SAMPLES], cutoff_bin)``."""
    paths = [[str(t.stems[s]) for s in cfg.STEMS] for t in tracks]
    nframes = [sf.info(p[0]).frames for p in paths]
    seg = cfg.SEG_SAMPLES

    def _load(idx):
        i = int(idx)
        rng = np.random.default_rng()  # bei jedem Zugriff neuer Crop/Mix
        n = nframes[i]
        start = 0 if n <= seg else int(rng.integers(0, n - seg + 1))
        length = min(seg, n)
        mode = str(rng.choice(DOWNMIX_MODES)) if augment else "mean"
        stems = {
            s: _read_crop_mono(p, start, length, mode)
            for s, p in zip(cfg.STEMS, paths[i])
        }
        target = augment_target(stems, rng=rng, enabled=augment)
        cb = int(rng.choice(cutoff_bins)) if len(cutoff_bins) > 1 else cutoff_bins[0]
        return target, np.int32(cb)

    return _load


# --------------------------------------------------------------------------- #
# Wellenform → Modell-Tensoren (TF-Graph)
# --------------------------------------------------------------------------- #
def waveform_to_tensors(target_wave: tf.Tensor, cutoff_bin=cfg.CUTOFF_BIN):
    """Vollband-Target-Wellenform → ``(input_spec[512,T,3], target_spec[512,T,2])``.

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

    input_spec.set_shape([cfg.N_BINS_NET, cfg.SEG_FRAMES, 3])
    target_spec.set_shape([cfg.N_BINS_NET, cfg.SEG_FRAMES, 2])
    return input_spec, target_spec


# --------------------------------------------------------------------------- #
# Öffentliche API
# --------------------------------------------------------------------------- #
def make_dataset(
    split: str,
    batch_size: int = 8,
    augment: bool | None = None,
    cutoffs=None,
    shuffle: bool = True,
    repeat: bool = True,
    seed: int = cfg.SEED,
) -> tf.data.Dataset:
    """tf.data.Dataset, der gepaarte Spektrogramme streamt.

    ``augment=None`` → an für 'train', aus für 'valid'/'test' (Test-Disziplin).
    ``cutoffs=None`` → ``cfg.CUTOFFS_HZ``; bei mehreren Werten zieht jeder Trainings-
    schritt zufällig einen, sonst/aus → der erste (deterministisch).
    """
    tracks = get_split(split)
    if not tracks:
        raise RuntimeError(f"Keine Tracks für Split {split!r} unter {cfg.DATA_ROOT}")
    if augment is None:
        augment = split == "train"
    cutoffs = cfg.CUTOFFS_HZ if cutoffs is None else cutoffs
    cutoff_bins = [cfg.cutoff_bin_for(hz) for hz in cutoffs]
    if not augment:
        cutoff_bins = cutoff_bins[:1]                # eval: fester (Standard-)Cutoff

    loader = _make_loader(tracks, augment, cutoff_bins)

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(tracks)))
    if shuffle:
        ds = ds.shuffle(len(tracks), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    def _map(idx):
        wave, cb = tf.numpy_function(
            loader, [idx], [tf.float32, tf.int32], name="load_target"
        )
        wave = tf.ensure_shape(wave, [cfg.SEG_SAMPLES])
        return waveform_to_tensors(wave, cb)

    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def sample_target(
    split: str, track_index: int = 0, augment: bool = False
) -> np.ndarray:
    """Ein einzelnes Target-Wellenform-Beispiel (für Notebook/Tests)."""
    tracks = get_split(split)
    loader = _make_loader(tracks, augment, [cfg.CUTOFF_BIN])
    wave, _ = loader(track_index)
    return wave
