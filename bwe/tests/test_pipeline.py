"""Schritt 6 — Split-Disjunktheit + tf.data-Pipeline (Shapes, Copy-Up, LF-Treue)."""

import numpy as np
import pytest

from bwe import config as cfg
from bwe.data import splits as SP
from bwe.data import pipeline as PL

# Pipeline braucht den 32-kHz-Cache; ohne ihn überspringen statt zu failen.
_HAS_DATA = bool(SP.get_split("train")) and bool(SP.get_split("test"))
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="32-kHz-Cache fehlt (prepare_sample.py)")


def test_splits_disjoint():
    names = {s: {t.name for t in SP.get_split(s)} for s in SP.SPLIT_NAMES}
    assert names["train"].isdisjoint(names["test"])
    assert names["train"].isdisjoint(names["valid"])
    assert names["valid"].isdisjoint(names["test"])
    # valid-Tracks sind genau die kanonischen Validation-Tracks.
    assert names["valid"] <= set(SP.VALIDATION_TRACKS)
    assert names["train"].isdisjoint(SP.VALIDATION_TRACKS)


def test_batch_shapes_and_finite():
    ds = PL.make_dataset("train", batch_size=2, augment=True, shuffle=False)
    inp, tgt = next(iter(ds.take(1)))
    assert tuple(inp.shape) == (2, cfg.N_BINS_NET, cfg.SEG_FRAMES, 3)
    assert tuple(tgt.shape) == (2, cfg.N_BINS_NET, cfg.SEG_FRAMES, 2)
    assert np.all(np.isfinite(inp.numpy())) and np.all(np.isfinite(tgt.numpy()))


def test_input_hf_filled_target_hf_real():
    ds = PL.make_dataset("train", batch_size=1, augment=False, shuffle=False)
    inp, tgt = next(iter(ds.take(1)))
    inp, tgt = inp.numpy()[0], tgt.numpy()[0]
    c = cfg.CUTOFF_BIN
    # Input-HF ist durch Copy-Up gefüllt (nicht ~0).
    hf_in = np.abs(inp[c:, :, :2]).mean()
    assert hf_in > 1e-3, hf_in
    # Target-HF ist echter Inhalt (> 0).
    hf_t = np.abs(tgt[c:, :, :]).mean()
    assert hf_t > 1e-4, hf_t


def test_input_lf_matches_target_lf():
    """LF stammt bei beiden aus derselben Wellenform → muss übereinstimmen (Bin-Ausrichtung)."""
    ds = PL.make_dataset("train", batch_size=1, augment=False, shuffle=False)
    inp, tgt = next(iter(ds.take(1)))
    inp, tgt = inp.numpy()[0], tgt.numpy()[0]
    c = cfg.CUTOFF_BIN
    sl = slice(4, -4)  # Randframes ausnehmen
    mag_in = np.hypot(inp[:c, sl, 0], inp[:c, sl, 1]).ravel()
    mag_t = np.hypot(tgt[:c, sl, 0], tgt[:c, sl, 1]).ravel()
    corr = np.corrcoef(mag_in, mag_t)[0, 1]
    rel = np.mean(np.abs(mag_in - mag_t)) / (np.mean(mag_t) + 1e-9)
    assert corr > 0.99, corr
    assert rel < 0.05, rel


def test_freq_coord_channel():
    ds = PL.make_dataset("train", batch_size=1, augment=False, shuffle=False)
    inp, _ = next(iter(ds.take(1)))
    freq = inp.numpy()[0, :, :, 2]
    expected = np.linspace(0.0, 1.0, cfg.N_BINS_NET)[:, None]
    assert np.allclose(freq, expected)            # über die Zeit konstant
    assert freq.min() == 0.0 and freq.max() == 1.0


def test_sample_target_is_mono():
    wave = PL.sample_target("train", track_index=0, augment=False)
    assert wave.ndim == 1
    assert wave.shape[0] == cfg.SEG_SAMPLES
    assert wave.dtype == np.float32


def test_downmix_modes():
    """mean/left/right liefern je ein Mono-Signal; L und R unterscheiden sich (Stereo)."""
    path = str(SP.get_split("train")[0].stems[cfg.STEMS[0]])
    for mode in PL.DOWNMIX_MODES:
        w = PL._read_crop_mono(path, 0, cfg.SEG_SAMPLES, mode)
        assert w.ndim == 1 and w.shape[0] == cfg.SEG_SAMPLES
    left = PL._read_crop_mono(path, 0, cfg.SEG_SAMPLES, "left")
    right = PL._read_crop_mono(path, 0, cfg.SEG_SAMPLES, "right")
    assert not np.array_equal(left, right)


def test_variable_cutoff_pipeline_runs():
    """Mehrere Cutoffs (variabler Cutoff pro Beispiel) laufen durch den Graph."""
    ds = PL.make_dataset("train", batch_size=2, augment=True,
                         cutoffs=(4000, 8000), shuffle=False)
    inp, tgt = next(iter(ds.take(1)))
    assert tuple(inp.shape) == (2, cfg.N_BINS_NET, cfg.SEG_FRAMES, 3)
    assert tuple(tgt.shape) == (2, cfg.N_BINS_NET, cfg.SEG_FRAMES, 2)
    assert np.all(np.isfinite(inp.numpy())) and np.all(np.isfinite(tgt.numpy()))
