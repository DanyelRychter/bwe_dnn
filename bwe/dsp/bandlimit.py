"""Bandbegrenzung per STFT-Brickwall.

Erzeugt aus einer Vollband-Wellenform den bandbegrenzten Input: STFT → alle Bins
ab ``cutoff_bin`` auf 0 → iSTFT. Bewusst dieselbe ``N_FFT``/``HOP`` wie die
Haupt-Pipeline, damit die Bin-Grenze *exakt* dort liegt, wo später Loss und
Splicing arbeiten (keine halbe Bin-Verschiebung).

Das harte Nullsetzen erzeugt theoretisch Gibbs-Ringing; die überlappenden
Hann-Frames + Overlap-Add dämpfen es so stark, dass es als „Vorher"-Demo
unauffällig ist (der hörbare Effekt ist das fehlende HF selbst).
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg
from bwe.dsp.stft import stft, istft


def bandlimit_spec(spec, cutoff_bin: int = cfg.CUTOFF_BIN) -> tf.Tensor:
    """Setzt alle HF-Bins (``>= cutoff_bin``) eines Spektrogramms ``[F, T]`` auf 0."""
    spec = tf.convert_to_tensor(spec, dtype=tf.complex64)
    n_bins = tf.shape(spec)[0]
    keep = tf.cast(tf.range(n_bins) < cutoff_bin, spec.dtype)  # [F]
    return spec * keep[:, tf.newaxis]


def bandlimit(wave, cutoff_bin: int = cfg.CUTOFF_BIN) -> tf.Tensor:
    """Vollband-Wellenform → bandbegrenzte Wellenform (Brickwall-Tiefpass)."""
    spec = stft(wave)
    spec = bandlimit_spec(spec, cutoff_bin)
    return istft(spec)
