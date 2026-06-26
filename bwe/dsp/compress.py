"""Power-Law-Kompression — nur auf der Magnitude, Phase exakt erhalten.

Idee: ``|S|^c · e^{jφ}``, gerechnet als ``S · |S|^(c-1)``. Bei ``c < 1`` werden
laute Bins gestaucht und leise angehoben — gleicht den riesigen Dynamikumfang aus
(StandardScaler-Analogon), ohne die Phase anzutasten. Invertierung mit ``1/c``.

Ein kleines Epsilon stabilisiert die Potenz bei ``|S| → 0`` (sonst NaN/Inf).
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg


def _scale(spec: tf.Tensor, exponent: float, eps: float) -> tf.Tensor:
    mag = tf.abs(spec)
    factor = tf.pow(mag + eps, exponent - 1.0)
    return spec * tf.cast(factor, spec.dtype)


def compress(spec, c: float = cfg.COMPRESS_C, eps: float = cfg.COMPRESS_EPS) -> tf.Tensor:
    """Komprimiert die Magnitude mit Exponent ``c`` (< 1 staucht laute Anteile)."""
    spec = tf.convert_to_tensor(spec, dtype=tf.complex64)
    return _scale(spec, c, eps)


def decompress(spec, c: float = cfg.COMPRESS_C, eps: float = cfg.COMPRESS_EPS) -> tf.Tensor:
    """Invertiert :func:`compress` (Exponent ``1/c``)."""
    spec = tf.convert_to_tensor(spec, dtype=tf.complex64)
    return _scale(spec, 1.0 / c, eps)
