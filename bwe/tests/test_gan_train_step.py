"""Schritt 15 — GAN-``train_step`` Smoke-Test (ein Step, kein Training).

Prüft Korrektheit des Loops, nicht Lernfähigkeit: endliche Metriken, beide Netze bewegen
sich, und der D-Vorlauf-Step lässt den Generator in Ruhe. Kleine Tensoren (T=16) → CPU-tauglich.
"""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.train.gan import build_model

T = 16


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.standard_normal((2, cfg.N_BINS_NET, T, cfg.N_INPUT_CHANNELS)).astype("float32")
    tgt = rng.standard_normal((2, cfg.N_BINS_NET, T, cfg.N_OUTPUT_CHANNELS)).astype("float32")
    return tf.constant(inp), tf.constant(tgt)


def test_train_step_runs_and_updates():
    """Ein train_step → endliche Metriken; G- und D-Gewichte ändern sich."""
    model = build_model()
    g_before = [v.numpy().copy() for v in model.generator.trainable_variables]
    d_before = [v.numpy().copy() for v in model.discriminator.trainable_variables]

    logs = model.train_step(_batch())
    for k in ("d_loss", "g_loss", "recon", "adv", "fm", "lsd_hf"):
        assert np.isfinite(float(logs[k])), k

    assert any(not np.allclose(a, b.numpy())
               for a, b in zip(g_before, model.generator.trainable_variables))
    assert any(not np.allclose(a, b.numpy())
               for a, b in zip(d_before, model.discriminator.trainable_variables))


def test_d_train_step_only_updates_d():
    """Diskriminator-Vorlauf: nur D wird aktualisiert, der Generator bleibt eingefroren."""
    model = build_model()
    g_before = [v.numpy().copy() for v in model.generator.trainable_variables]
    d_before = [v.numpy().copy() for v in model.discriminator.trainable_variables]

    model.d_train_step(_batch(1))

    assert all(np.allclose(a, b.numpy())
               for a, b in zip(g_before, model.generator.trainable_variables))
    assert any(not np.allclose(a, b.numpy())
               for a, b in zip(d_before, model.discriminator.trainable_variables))


def test_test_step_reports_quality_metrics():
    """Val-Step liefert endliches lsd_hf + recon (Generator-Qualität)."""
    model = build_model()
    logs = model.test_step(_batch(2))
    assert np.isfinite(float(logs["lsd_hf"]))
    assert np.isfinite(float(logs["recon"]))
