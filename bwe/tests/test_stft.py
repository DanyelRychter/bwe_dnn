"""Schritt 2 — STFT/iSTFT-Round-Trip und Nyquist-Handling."""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.dsp import stft as S


def _demo_signal(n=cfg.SEG_SAMPLES, seed=0):
    """Deterministisches Breitband-Signal (Summe einiger Sinus + leichtes Rauschen)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / cfg.SR
    x = np.zeros(n, dtype=np.float32)
    for f in (220.0, 1000.0, 5000.0, 9000.0):
        x += np.sin(2 * np.pi * f * t).astype(np.float32)
    x += 0.01 * rng.standard_normal(n).astype(np.float32)
    return x / np.max(np.abs(x))


def test_shapes():
    spec = S.stft(_demo_signal())
    assert spec.shape[0] == cfg.N_BINS_FULL == 513
    assert spec.shape[1] == cfg.SEG_FRAMES == 128
    assert spec.dtype == tf.complex64


def test_round_trip_interior():
    x = _demo_signal()
    y = S.istft(S.stft(x)).numpy()
    # Längen identisch, wenn das Signal das Raster genau füllt (SEG_SAMPLES).
    assert len(y) == len(x)
    pad = cfg.N_FFT  # Randframes (partielle Überlappung) ausnehmen
    err = np.max(np.abs(y[pad:-pad] - x[pad:-pad]))
    assert err < 1e-4, f"Round-Trip-Fehler im Inneren zu groß: {err}"


def test_drop_pad_nyquist():
    spec = S.stft(_demo_signal())
    net = S.drop_nyquist(spec)
    assert net.shape[0] == cfg.N_BINS_NET == 512
    back = S.pad_nyquist(net)
    assert back.shape[0] == cfg.N_BINS_FULL
    # Alle Bins außer dem (genullten) Nyquist-Bin unverändert.
    diff = np.max(np.abs(back.numpy()[:512] - spec.numpy()[:512]))
    assert diff == 0.0
    assert np.all(back.numpy()[512] == 0)


def test_drop_pad_round_trip_is_clean():
    """drop→pad→iSTFT bleibt nahezu identisch (Nyquist energiearm)."""
    x = _demo_signal()
    spec = S.stft(x)
    y = S.istft(S.pad_nyquist(S.drop_nyquist(spec))).numpy()
    pad = cfg.N_FFT
    err = np.max(np.abs(y[pad:-pad] - x[pad:-pad]))
    assert err < 1e-3, f"drop/pad-Round-Trip zu ungenau: {err}"
