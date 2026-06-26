"""Schritt 3 — Power-Law-Kompression: Round-Trip, Phasenerhalt, Null-Stabilität."""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.dsp import compress as C


def _demo_spec(seed=0):
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((513, 64)).astype(np.float32)
    im = rng.standard_normal((513, 64)).astype(np.float32)
    # gemischte Größenordnungen (großer Dynamikumfang)
    scale = (10.0 ** rng.uniform(-3, 3, size=(513, 1))).astype(np.float32)
    return tf.constant((re + 1j * im) * scale, dtype=tf.complex64)


def test_round_trip():
    s = _demo_spec()
    back = C.decompress(C.compress(s)).numpy()
    rel = np.abs(back - s.numpy()) / (np.abs(s.numpy()) + 1e-6)
    assert np.nanmax(rel) < 1e-3, f"Kompressions-Round-Trip ungenau: {np.nanmax(rel)}"


def test_phase_preserved():
    s = _demo_spec()
    comp = C.compress(s).numpy()
    s_np = s.numpy()
    # Phase nur dort prüfen, wo Magnitude nicht winzig ist.
    mask = np.abs(s_np) > 1e-4
    dphi = np.angle(comp[mask]) - np.angle(s_np[mask])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(dphi)) < 1e-4


def test_zero_bins_no_nan():
    s = tf.zeros((513, 16), dtype=tf.complex64)
    comp = C.compress(s).numpy()
    back = C.decompress(C.compress(s)).numpy()
    assert not np.any(np.isnan(comp)) and not np.any(np.isnan(back))
    assert np.all(comp == 0) and np.all(back == 0)


def test_compression_reduces_dynamic_range():
    s = _demo_spec()
    comp = C.compress(s).numpy()
    # Variationskoeffizient der Magnitude sinkt deutlich.
    cv_in = np.std(np.abs(s.numpy())) / np.mean(np.abs(s.numpy()))
    cv_out = np.std(np.abs(comp)) / np.mean(np.abs(comp))
    assert cv_out < cv_in
