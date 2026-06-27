"""Metriken zur Bewertung (nicht zum Optimieren — müssen nicht differenzierbar sein).

* **LSD-HF** (Hauptzahl): Log-Spectral Distance, aber nur über die HF-Bins gerechnet
  (das LF ist bei allen Modellen identisch durchgesplict → würde den Vergleich
  verwässern). RMS der Differenz der Log-Magnitudenspektren, pro Frame gemittelt.
* **SI-SDR**: Scale-Invariant SDR auf der Wellenform. Bestraft die halluzinierte
  Phase generativer Modelle — die Divergenz zur LSD/zum Höreindruck ist die Pointe
  der Präsentation.

Beide arbeiten auf Wellenformen und richten unterschiedliche Längen vorher aus.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg
from bwe.dsp.stft import stft


def _log10(x: tf.Tensor) -> tf.Tensor:
    return tf.math.log(x) / tf.math.log(tf.constant(10.0, x.dtype))


def _align(a: tf.Tensor, b: tf.Tensor):
    """Auf gemeinsame (minimale) Länge croppen — Rekonstruktion kann minimal kürzer sein."""
    n = tf.minimum(tf.shape(a)[0], tf.shape(b)[0])
    return a[:n], b[:n]


def lsd_hf(
    pred_wave,
    target_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    eps: float = 1e-8,
) -> float:
    """Log-Spectral Distance über die HF-Bins (in dB; kleiner = besser)."""
    pred = tf.convert_to_tensor(pred_wave, tf.float32)
    target = tf.convert_to_tensor(target_wave, tf.float32)
    pred, target = _align(pred, target)

    log_p = 20.0 * _log10(tf.abs(stft(pred)) + eps)      # [513, T]
    log_t = 20.0 * _log10(tf.abs(stft(target)) + eps)
    sq = tf.square(log_p - log_t)[cutoff_bin:]           # nur HF
    per_frame = tf.sqrt(tf.reduce_mean(sq, axis=0))      # RMS über Frequenz -> [T]
    return float(tf.reduce_mean(per_frame))


def si_sdr(pred_wave, target_wave, eps: float = 1e-8) -> float:
    """Scale-Invariant SDR in dB (größer = besser). Invariant gegen Skalierung von pred."""
    pred = tf.convert_to_tensor(pred_wave, tf.float32)
    target = tf.convert_to_tensor(target_wave, tf.float32)
    pred, target = _align(pred, target)
    pred = pred - tf.reduce_mean(pred)
    target = target - tf.reduce_mean(target)

    alpha = tf.reduce_sum(pred * target) / (tf.reduce_sum(target * target) + eps)
    proj = alpha * target                                # Projektion von pred auf target
    noise = pred - proj
    ratio = tf.reduce_sum(proj * proj) / (tf.reduce_sum(noise * noise) + eps)
    return float(10.0 * _log10(ratio + eps))
