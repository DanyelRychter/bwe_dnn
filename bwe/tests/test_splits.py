"""Split-Zusammenstellung (bwe.data.splits) — insbesondere der Cache-Integritätscheck."""

import numpy as np
import pytest
import soundfile as sf

from bwe import config as cfg
from bwe.data import splits as SP


def _write_track(root, subset, name, frames_per_stem):
    """Track-Ordner mit Stem-WAVs anlegen; ``frames_per_stem`` = {stem: n_frames}."""
    d = root / subset / name
    d.mkdir(parents=True)
    for stem, n in frames_per_stem.items():
        sf.write(str(d / f"{stem}.wav"), np.zeros(n, dtype=np.float32), cfg.SR)


def test_get_split_skips_length_mismatched_track(tmp_path, capsys):
    good = {s: 1000 for s in cfg.STEMS}
    bad = {**good, "vocals": 500}                      # ein Stem kürzer = Cache-Defekt
    _write_track(tmp_path, "test", "Good Track", good)
    _write_track(tmp_path, "test", "Bad Track", bad)

    tracks = SP.get_split("test", root=tmp_path)

    assert [t.name for t in tracks] == ["Good Track"]
    assert "Bad Track" in capsys.readouterr().out           # Warnung ausgegeben


def test_get_split_skips_incomplete_track(tmp_path):
    good = {s: 1000 for s in cfg.STEMS}
    incomplete = {s: 1000 for s in cfg.STEMS if s != "bass"}
    _write_track(tmp_path, "train", "Complete Track", good)
    _write_track(tmp_path, "train", "Incomplete Track", incomplete)

    tracks = SP.get_split("train", root=tmp_path)

    assert [t.name for t in tracks] == ["Complete Track"]


def test_describe_split_columns_and_duration(tmp_path):
    _write_track(tmp_path, "test", "Track A", {s: cfg.SR for s in cfg.STEMS})  # 1 s

    df = SP.describe_split("test", root=tmp_path)

    assert list(df.columns) == ["split", "index", "track", "dauer_s", "samples", "sr_hz"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["track"] == "Track A"
    assert row["samples"] == cfg.SR
    assert row["dauer_s"] == pytest.approx(1.0)
    assert row["sr_hz"] == cfg.SR
