"""Demo-Audio-Lader (bwe.data.loaders)."""

import numpy as np
import pytest

from bwe import config as cfg
from bwe.data import loaders as L
from bwe.data import splits as SP

_HAS_DATA = bool(SP.get_split("train"))
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="32-kHz-Cache fehlt (prepare_sample.py)")


def test_load_demo_mix_mono_length_and_norm():
    name, wave = L.load_demo("train", index=0, seconds=2.0, offset=5.0)
    assert isinstance(name, str) and name
    assert wave.ndim == 1 and wave.dtype == np.float32
    assert wave.shape[0] == int(2.0 * cfg.SR)
    assert np.max(np.abs(wave)) <= 1.0 + 1e-6              # normiert


def test_load_demo_single_stem_differs_from_mix():
    _, mix = L.load_demo("train", 0, seconds=2.0, offset=5.0, stems="mix", normalize=False)
    _, drums = L.load_demo("train", 0, seconds=2.0, offset=5.0, stems="drums", normalize=False)
    assert mix.shape == drums.shape
    assert not np.array_equal(mix, drums)


def test_load_demo_combination_sums_stems():
    kw = dict(seconds=1.0, offset=0.0, normalize=False)
    _, drums = L.load_demo("train", 0, stems="drums", **kw)
    _, bass = L.load_demo("train", 0, stems="bass", **kw)
    _, both = L.load_demo("train", 0, stems=("drums", "bass"), **kw)
    assert np.allclose(both, drums + bass, atol=1e-5)


def test_load_demo_bad_stem_raises():
    with pytest.raises(ValueError):
        L.load_demo("train", 0, stems="guitar")


def test_load_demo_bad_split_raises():
    with pytest.raises(ValueError):
        L.load_demo("nope", 0)
