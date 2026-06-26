"""Schritt 4 — Bandbegrenzung (STFT-Brickwall)."""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.dsp import stft as S
from bwe.dsp import bandlimit as B


def _two_tone(n=cfg.SEG_SAMPLES):
    """LF-Ton (2 kHz, unter Cutoff) + HF-Ton (12 kHz, über Cutoff)."""
    t = np.arange(n) / cfg.SR
    lf = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    hf = np.sin(2 * np.pi * 12000 * t).astype(np.float32)
    return lf, hf, (lf + hf)


def test_bandlimit_spec_zeros_hf_exactly():
    spec = tf.ones((513, 32), dtype=tf.complex64)
    out = B.bandlimit_spec(spec).numpy()
    assert np.all(out[cfg.CUTOFF_BIN:] == 0)
    assert np.all(out[: cfg.CUTOFF_BIN] == 1)


def test_hf_removed_lf_kept():
    lf, hf, mix = _two_tone()
    y = B.bandlimit(mix).numpy()
    spec = np.abs(S.stft(y).numpy())
    hf_energy = spec[cfg.CUTOFF_BIN:].sum()
    lf_energy = spec[: cfg.CUTOFF_BIN].sum()
    # HF praktisch verschwunden gegenüber LF.
    assert hf_energy / lf_energy < 1e-3, hf_energy / lf_energy
    # LF-Inhalt im Inneren erhalten (12-kHz-Ton ist weg, 2-kHz-Ton bleibt).
    pad = cfg.N_FFT
    corr = np.corrcoef(y[pad:-pad], lf[pad:-pad])[0, 1]
    assert corr > 0.95, corr
