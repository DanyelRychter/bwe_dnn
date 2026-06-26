"""Schritt 5 — Copy-Up (HF-Initialisierung per Hochkopie)."""

import numpy as np
import tensorflow as tf

from bwe import config as cfg
from bwe.dsp import copyup as CU


def _lf_only_spec(seed=0, n_bins=cfg.N_BINS_NET, t=64):
    """Spektrogramm mit gefülltem LF (0..Cutoff) und leerem HF."""
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((n_bins, t)).astype(np.float32)
    im = rng.standard_normal((n_bins, t)).astype(np.float32)
    spec = (re + 1j * im).astype(np.complex64)
    spec[cfg.CUTOFF_BIN:] = 0  # HF leer
    return tf.constant(spec)


def test_hf_filled_lf_unchanged():
    spec = _lf_only_spec()
    out = CU.copy_up_hf(spec, energy_match=False).numpy()
    # LF unverändert.
    assert np.array_equal(out[: cfg.CUTOFF_BIN], spec.numpy()[: cfg.CUTOFF_BIN])
    # HF jetzt gefüllt (war vorher 0).
    assert np.all(np.abs(out[cfg.CUTOFF_BIN:]).sum(axis=0) > 0)


def test_tiling_matches_source_band():
    spec = _lf_only_spec()
    out = CU.copy_up_hf(spec, energy_match=False).numpy()
    lo, hi = cfg.COPYUP_SRC_LO_BIN, cfg.COPYUP_SRC_HI_BIN  # 128, 256
    width = hi - lo                                         # 128
    src = spec.numpy()[lo:hi]
    # Erste Kachel (8–12 kHz) = Quellband; zweite Kachel (12–16 kHz) ebenfalls.
    tile1 = out[cfg.CUTOFF_BIN: cfg.CUTOFF_BIN + width]
    tile2 = out[cfg.CUTOFF_BIN + width: cfg.CUTOFF_BIN + 2 * width]
    assert np.allclose(tile1, src)
    assert np.allclose(tile2, src)


def test_energy_match_changes_crossover_level():
    spec = _lf_only_spec()
    out_raw = CU.copy_up_hf(spec, energy_match=False).numpy()
    out_em = CU.copy_up_hf(spec, energy_match=True).numpy()
    rw = 16
    ref = np.abs(spec.numpy()[cfg.CUTOFF_BIN - rw: cfg.CUTOFF_BIN]).mean()
    # Mit Energieabgleich liegt der Pegel direkt über dem Cutoff näher am LF-Rand.
    lvl_raw = np.abs(out_raw[cfg.CUTOFF_BIN: cfg.CUTOFF_BIN + rw]).mean()
    lvl_em = np.abs(out_em[cfg.CUTOFF_BIN: cfg.CUTOFF_BIN + rw]).mean()
    assert abs(lvl_em - ref) < abs(lvl_raw - ref)


def test_works_on_full_513_spec():
    """Generisch auch auf vollem 513-Bin-Spektrogramm (krummes n_hf=257)."""
    spec = _lf_only_spec(n_bins=cfg.N_BINS_FULL)
    out = CU.copy_up_hf(spec, energy_match=True).numpy()
    assert out.shape[0] == cfg.N_BINS_FULL
    assert np.all(np.abs(out[cfg.CUTOFF_BIN:]).sum(axis=0) > 0)


def test_variable_cutoff_generalizes():
    """Mit anderem Cutoff (4 kHz = Bin 128): Quellband = Oktave darunter (2–4 kHz)."""
    spec = _lf_only_spec()                     # Bins 0..255 gefüllt
    cb = cfg.cutoff_bin_for(4000)              # 128
    out = CU.copy_up_hf(spec, energy_match=False, cutoff_bin=cb).numpy()
    assert np.array_equal(out[:cb], spec.numpy()[:cb])        # LF (< 4 kHz) unverändert
    assert np.all(np.abs(out[cb:]).sum(axis=0) > 0)           # HF gefüllt
    width = cb // 2                                            # Quellband cb//2..cb = 64..128
    src = spec.numpy()[cb - width:cb]
    assert np.allclose(out[cb:cb + width], src)               # erste Kachel = Quellband
