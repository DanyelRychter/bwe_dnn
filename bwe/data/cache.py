"""Einmaliges 32-kHz-Caching der Stems — gegen die I/O-/CPU-Bound-Pipeline (Kaggle).

Die ``tf.data``-Pipeline resampelt sonst jeden Crop on-the-fly von 44,1 kHz auf
``cfg.SR`` (CPU) → die GPU hungert (im Subset-Lauf nur ~17 % ausgelastet). Hier
resampeln wir die Stems **aller** ``train/``- und ``test/``-Tracks **einmal** in einen
Cache. Zeigt ``cfg.DATA_ROOT`` danach auf den Cache, gilt im Training ``sr == cfg.SR``
→ der Resample-Zweig der Pipeline entfällt → GPU-Bound.

Bewusst wiederverwendbar gehalten (auch fürs GAN-Notebook etc.):
* :func:`build_resample_cache` — resampelt und gibt den Cache-Pfad zurück.
* :func:`use_resample_cache` — baut den Cache **und** biegt ``cfg.DATA_ROOT`` darauf.

Eigenschaften:
* **Stereo bleibt erhalten** → die Kanal-Augmentation (mean/left/right) in
  :mod:`bwe.data.pipeline` funktioniert unverändert.
* **``mixture.wav`` wird nicht gecacht** (Mixture = Summe der Stems) → spart Platz/Zeit.
* **Resumebar:** bereits gecachte Dateien werden übersprungen.
* ``track.subset`` spiegelt die physische ``train/``-/``test/``-Struktur → der
  kanonische 86/14/50-Split reproduziert sich aus dem Cache (valid liegt in ``train/``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import soxr

from bwe import config as cfg
from bwe.data import splits as SP


def build_resample_cache(
    src_root,
    cache_root,
    *,
    sr: int = cfg.SR,
    stems: tuple[str, ...] = cfg.STEMS,
    quality: str = "HQ",
    verbose: bool = True,
) -> Path:
    """Resampelt alle Stems von ``src_root`` nach ``cache_root`` (gibt ``cache_root`` zurück).

    ``quality="HQ"`` entspricht dem ``soxr_hq`` der Pipeline. ``src_root`` muss
    ``train/`` und ``test/`` enthalten; der Cache erhält dieselbe Struktur.
    """
    src_root, cache_root = Path(src_root), Path(cache_root)
    tracks = (
        SP.get_split("train", root=src_root)
        + SP.get_split("valid", root=src_root)
        + SP.get_split("test", root=src_root)
    )
    if not tracks:
        raise RuntimeError(f"Keine vollständigen Tracks unter {src_root} gefunden.")
    if verbose:
        print(f"32k-Cache: {len(tracks)} Tracks x {len(stems)} Stems  {src_root} -> {cache_root}")

    t0 = time.time()
    for i, track in enumerate(tracks, 1):
        out_dir = cache_root / track.subset / track.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            dst = out_dir / f"{stem}.wav"
            if dst.exists():                       # resumebar: schon gecacht -> überspringen
                continue
            src = track.stems[stem]
            info = sf.info(str(src))
            data, native_sr = sf.read(str(src), always_2d=True, dtype="float32")
            if native_sr != sr:
                data = soxr.resample(data, native_sr, sr, quality=quality)
            sf.write(str(dst), data, sr, subtype=info.subtype or "PCM_16")
        if verbose and (i % 10 == 0 or i == len(tracks)):
            print(f"  [{i:3d}/{len(tracks)}] {time.time() - t0:6.0f}s  {track.name[:42]}")

    if verbose:
        print(f"Cache fertig in {time.time() - t0:.0f}s.")
    return cache_root


def use_resample_cache(src_root, cache_root="/kaggle/working/musdb18hq_32k", **kw) -> Path:
    """Cache bauen **und** ``cfg.DATA_ROOT`` darauf umbiegen (Pipeline liest dann 32k direkt).

    Wichtig: ``cfg.DATA_ROOT`` wird direkt gesetzt — ``os.environ`` allein wirkt nach dem
    ``bwe.config``-Import nicht mehr (der Wert wurde beim Import gelesen).
    """
    cache_root = build_resample_cache(src_root, cache_root, **kw)
    cfg.DATA_ROOT = cache_root
    os.environ["BWE_DATA_ROOT"] = str(cache_root)
    if kw.get("verbose", True):
        print(f"cfg.DATA_ROOT -> {cfg.DATA_ROOT}")
    return cache_root
