"""Schritt 10 — RI+Mag-Loss (HF-maskiert) und Splicing."""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.losses import ri_mag_loss, splice

F, T = cfg.N_BINS_NET, 16


def _rand(seed, ch=2):
    rng = np.random.default_rng(seed)
    return tf.constant(rng.standard_normal((2, F, T, ch)).astype("float32"))


def test_loss_zero_when_equal():
    x = _rand(0)
    assert float(ri_mag_loss(x, x)) < 1e-6


def test_loss_positive_when_different():
    assert float(ri_mag_loss(_rand(0), _rand(1))) > 0.0


def test_loss_ignores_lf():
    """Unterschied nur im LF → Loss bleibt 0 (es zählt nur das HF)."""
    t = _rand(2)
    p = t.numpy().copy()
    p[:, : cfg.CUTOFF_BIN] += 5.0
    assert float(ri_mag_loss(tf.constant(p), t)) < 1e-6


def test_gradient_only_in_hf():
    t = _rand(3)
    p = tf.Variable(_rand(4))
    with tf.GradientTape() as tape:
        loss = ri_mag_loss(p, t)
    g = tape.gradient(loss, p).numpy()
    assert np.allclose(g[:, : cfg.CUTOFF_BIN], 0.0)              # kein Gradient im LF
    assert np.abs(g[:, cfg.CUTOFF_BIN:]).sum() > 0.0             # Gradient im HF


def test_splice_keeps_lf_uses_hf():
    model_out = _rand(5, ch=2)
    input_spec = _rand(6, ch=3)                                  # Re, Im, Freq-Koord
    out = splice(model_out, input_spec).numpy()
    c = cfg.CUTOFF_BIN
    assert np.array_equal(out[:, :c], input_spec.numpy()[:, :c, :, :2])
    assert np.array_equal(out[:, c:], model_out.numpy()[:, c:])


def test_splice_gradient_hf_only():
    input_spec = _rand(7, ch=3)
    model_out = tf.Variable(_rand(8, ch=2))
    with tf.GradientTape() as tape:
        s = tf.reduce_sum(splice(model_out, input_spec))
    g = tape.gradient(s, model_out).numpy()
    c = cfg.CUTOFF_BIN
    assert np.allclose(g[:, :c], 0.0)                            # LF hängt nicht von model_out ab
    assert np.allclose(g[:, c:], 1.0)                            # HF 1:1 durchgereicht
