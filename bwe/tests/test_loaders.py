"""Demo-Audio-Lader (bwe.data.loaders)."""

import numpy as np
import pytest

from bwe import config as cfg
from bwe.data import loaders as L
from bwe.data import splits as SP

_HAS_DATA = bool(SP.get_split("train"))
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="32-kHz-Cache fehlt (prepare_sample.py)")


def test_load_mix_mono_length_and_norm():
    track = SP.get_split("train")[0]
    wave = L.load_mix(track, seconds=2.0, offset=5.0)
    assert wave.ndim == 1
    assert wave.dtype == np.float32
    assert wave.shape[0] == int(2.0 * cfg.SR)
    assert np.max(np.abs(wave)) <= 1.0 + 1e-6          # normiert


def test_load_demo_returns_name_and_wave():
    name, wave = L.load_demo("train", index=0, seconds=1.0, offset=0.0)
    assert isinstance(name, str) and name
    assert wave.ndim == 1 and wave.shape[0] == int(1.0 * cfg.SR)
