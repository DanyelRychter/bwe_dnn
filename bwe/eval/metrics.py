"""Metriken zur Bewertung (nicht zum Optimieren — müssen nicht differenzierbar sein).

* **LSD-HF** (Hauptzahl): Log-Spectral Distance, aber nur über die HF-Bins gerechnet
  (das LF ist bei allen Modellen identisch durchgesplict → würde den Vergleich
  verwässern). RMS der Differenz der Log-Magnitudenspektren, pro Frame gemittelt.
* **SI-SDR**: Scale-Invariant SDR auf der Wellenform. Bestraft die halluzinierte
  Phase generativer Modelle — die Divergenz zur LSD/zum Höreindruck ist die Pointe
  der Präsentation.
* **SI-SDR-HF**: dasselbe, aber nach Hochpass beider Signale. Das Full-band-SI-SDR
  wird vom identisch geteilten Real-LF dominiert und ist damit HF-blind; erst der
  Hochpass macht die HF-Phasentreue sichtbar (Leitfaden §5.5).

Alle arbeiten auf Wellenformen und richten unterschiedliche Längen vorher aus.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg
from bwe.dsp.stft import istft, stft


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


def lsd_hf_spec(pred_spec, target_spec, cutoff_bin: int = cfg.CUTOFF_BIN,
                c: float = cfg.COMPRESS_C, eps: float = 1e-8) -> tf.Tensor:
    """LSD-HF direkt aus **komprimierten** Re/Im-Spektren ``[...,F,T,2]`` (in dB).

    Schnelle Trainings-/Val-Metrik ohne iSTFT. Die komprimierte Magnitude wird erst
    **entkomprimiert** (``|S| = |S_c|^(1/c)``) und dann wie bei :func:`lsd_hf`
    (Wellenform) als ``20·log10`` gemessen — so ist die Zahl mit der Wellenform-Metrik
    vergleichbar (frühere Variante blähte über ``20/c`` den eps-Boden leiser HF-Bins auf).
    Gibt einen Skalar-Tensor zurück (für Keras-Metriken)."""
    pred_spec = tf.convert_to_tensor(pred_spec)
    target_spec = tf.cast(target_spec, pred_spec.dtype)
    mp = tf.sqrt(tf.square(pred_spec[..., 0]) + tf.square(pred_spec[..., 1]))
    mt = tf.sqrt(tf.square(target_spec[..., 0]) + tf.square(target_spec[..., 1]))
    mp = tf.pow(mp, 1.0 / c)                                          # zurück zur Roh-Magnitude
    mt = tf.pow(mt, 1.0 / c)
    sq = tf.square(20.0 * _log10(mp + eps) - 20.0 * _log10(mt + eps))  # [..., F, T]
    m = _hf_mask_col(tf.shape(sq)[-2], cutoff_bin, sq.dtype)          # [F, 1]
    per_frame = tf.sqrt(tf.reduce_sum(sq * m, axis=-2) / (tf.reduce_sum(m) + eps))
    return tf.reduce_mean(per_frame)


def _hf_mask_col(n_bins, cutoff_bin, dtype):
    return tf.cast(tf.range(n_bins) >= cutoff_bin, dtype)[:, tf.newaxis]   # [F, 1]


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


def _highpass(wave, cutoff_bin: int):
    """Wellenform → nur HF behalten (``>= cutoff_bin``); komplementär zu ``bandlimit``."""
    spec = stft(wave)                                    # [513, T] complex
    keep = tf.cast(tf.range(tf.shape(spec)[0]) >= cutoff_bin, spec.dtype)
    return istft(spec * keep[:, tf.newaxis])


def si_sdr_hf(
    pred_wave,
    target_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    eps: float = 1e-8,
) -> float:
    """SI-SDR **nach Hochpass** beider Signale (in dB; größer = besser).

    Schärferer HF-Sanity-Check: das Full-band-SI-SDR wird vom bit-genau geteilten
    Real-LF dominiert (LF-Fehler ≈ 0, Energie LF-lastig) → HF-blind. Nach dem
    Hochpass bleibt nur das rekonstruierte HF übrig, sodass die dekorrelierte,
    halluzinierte GAN-Phase durchschlägt (Leitfaden §5.5).
    """
    return si_sdr(_highpass(pred_wave, cutoff_bin), _highpass(target_wave, cutoff_bin), eps)
