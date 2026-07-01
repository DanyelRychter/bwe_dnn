"""Schritt 8 — Metriken LSD-HF und SI-SDR."""

import numpy as np

from bwe import config as cfg
from bwe.eval import metrics as M


def _sig(n=cfg.SEG_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / cfg.SR
    x = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 9000 * t)
    x += 0.01 * rng.standard_normal(n)
    return x.astype(np.float32)


def test_lsd_zero_for_identical():
    x = _sig()
    assert M.lsd_hf(x, x) < 1e-3


def test_lsd_increases_for_worse():
    x = _sig()
    rng = np.random.default_rng(1)
    noisy = (x + 0.5 * rng.standard_normal(len(x))).astype(np.float32)
    assert M.lsd_hf(noisy, x) > M.lsd_hf(x, x)


def test_si_sdr_high_for_identical():
    x = _sig()
    assert M.si_sdr(x, x) > 100


def test_si_sdr_scale_invariant():
    """SI-SDR ist invariant gegen Skalierung des (imperfekten) Schätzers."""
    x = _sig()
    rng = np.random.default_rng(3)
    pred = (x + 0.3 * rng.standard_normal(len(x))).astype(np.float32)
    assert abs(M.si_sdr(pred, x) - M.si_sdr(5.0 * pred, x)) < 1e-2


def test_si_sdr_worse_for_noise():
    x = _sig()
    rng = np.random.default_rng(2)
    noisy = (x + 0.5 * rng.standard_normal(len(x))).astype(np.float32)
    assert M.si_sdr(noisy, x) < M.si_sdr(x, x)


def test_si_sdr_hf_finite_and_high_for_identical():
    x = _sig()
    v = M.si_sdr_hf(x, x)
    assert np.isfinite(v) and v > 50                 # identisches HF -> sehr hoch


def test_si_sdr_hf_worse_for_noise():
    x = _sig()
    rng = np.random.default_rng(5)
    noisy = (x + 0.2 * rng.standard_normal(len(x))).astype(np.float32)
    assert M.si_sdr_hf(noisy, x) < M.si_sdr_hf(x, x)
