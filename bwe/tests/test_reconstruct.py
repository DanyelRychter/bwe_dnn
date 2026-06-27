"""Schritt 7 — Copy-Up-Baseline End-to-End."""

import numpy as np

from bwe import config as cfg
from bwe.dsp.stft import stft
from bwe.eval import metrics as M
from bwe.infer import reconstruct as R


def _music(n=cfg.SEG_SAMPLES * 3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / cfg.SR
    x = np.zeros(n, dtype=np.float32)
    for f in (220, 440, 880, 3000, 6000, 11000):
        x += np.sin(2 * np.pi * f * t).astype(np.float32)
    x += 0.01 * rng.standard_normal(n).astype(np.float32)
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_reconstruct_fills_hf():
    x = _music()
    pred, inp = R.baseline_from_fullband(x)
    hf_in = np.abs(stft(inp).numpy())[cfg.CUTOFF_BIN:].mean()
    hf_pred = np.abs(stft(pred).numpy())[cfg.CUTOFF_BIN:].mean()
    assert hf_pred > 10 * hf_in                      # HF vorher ~leer, nachher gefüllt


def test_reconstruct_preserves_lf():
    x = _music()
    pred, inp = R.baseline_from_fullband(x)
    n = min(len(pred), len(inp))
    P = np.abs(stft(pred[:n]).numpy())[: cfg.CUTOFF_BIN]
    I = np.abs(stft(inp[:n]).numpy())[: cfg.CUTOFF_BIN]
    rel = np.abs(P[:, 3:-3] - I[:, 3:-3]).mean() / (I[:, 3:-3].mean() + 1e-9)
    assert rel < 0.05                                # LF bit-genau gesplict (bis auf Rand)


def test_copyup_beats_bandlimited_input():
    x = _music()
    pred, inp = R.baseline_from_fullband(x)
    assert M.lsd_hf(pred, x) < M.lsd_hf(inp, x)      # Copy-Up verbessert LSD-HF
