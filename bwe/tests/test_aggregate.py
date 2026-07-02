"""Schritt 17b — Aggregat-Auswertung über einen Split (bwe.eval.aggregate).

Datengebunden: braucht den 32-kHz-Cache des ``valid``-Splits.
"""

import numpy as np
import pytest

from bwe.data import splits as SP
from bwe.eval import aggregate as A

_HAS_VALID = bool(SP.get_split("valid"))
pytestmark = pytest.mark.skipif(not _HAS_VALID, reason="valid-Cache fehlt (32 kHz)")


class _IdGen:
    """Dummy-Generator ohne Training: gibt die RI-Kanäle des Eingangs zurück.

    Reicht, um den Modell-Pfad (``model_from_fullband`` → splice → istft) zu prüfen,
    ohne Gewichte laden zu müssen.
    """

    def __call__(self, x, training=False):
        return x[..., :2]


def test_evaluate_split_columns_and_rows():
    df = A.evaluate_split("valid", generators={"Model": _IdGen()},
                          limit=1, seconds=1.0, verbose=False)
    assert {"track", "method", "lsd_hf", "si_sdr", "si_sdr_hf"}.issubset(df.columns)
    assert set(df["method"]) == {"Copy-Up", "Model"}
    assert len(df) == 2                                      # 1 Track × 2 Methoden
    vals = df[["lsd_hf", "si_sdr", "si_sdr_hf"]].to_numpy(dtype=float)
    assert np.isfinite(vals).all()


def test_evaluate_split_without_hf_sisdr():
    df = A.evaluate_split("valid", limit=1, seconds=1.0, hf_sisdr=False, verbose=False)
    assert "si_sdr_hf" not in df.columns                     # optionale Metrik weglassbar
    assert list(df["method"]) == ["Copy-Up"]                 # ohne Generatoren nur Copy-Up


def test_summary_shape():
    df = A.evaluate_split("valid", limit=2, seconds=1.0, verbose=False)
    s = A.summary(df)
    assert list(s.index) == ["Copy-Up"]
    assert "LSD-HF [dB]" in s.columns
    assert isinstance(s.loc["Copy-Up", "LSD-HF [dB]"], str)  # "mean ± std"


def test_evaluate_split_include_input():
    df = A.evaluate_split("valid", limit=1, seconds=1.0, include_input=True,
                          hf_sisdr=False, verbose=False)
    assert list(df["method"]) == ["Bandbegrenzt", "Copy-Up"]
    # leeres HF-Band → LSD-HF des bandbegrenzten Signals ist riesig (eps-Boden)
    lsd = df.set_index("method")["lsd_hf"]
    assert lsd["Bandbegrenzt"] > lsd["Copy-Up"]


def test_compare_splits_columns():
    df = A.evaluate_split("valid", limit=2, seconds=1.0, hf_sisdr=False, verbose=False)
    tbl = A.compare_splits({"a": df, "b": df})
    # Metrik-major: beide Splits einer Metrik nebeneinander, Splits in Eingabereihenfolge
    assert list(tbl.columns) == [("LSD-HF [dB]", "a"), ("LSD-HF [dB]", "b"),
                                 ("SI-SDR [dB]", "a"), ("SI-SDR [dB]", "b")]
    assert list(tbl.index) == ["Copy-Up"]
