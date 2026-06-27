"""Audio-Lader für Notebooks/Demos — bewusst getrennt von der tf.data-Pipeline.

Ein Track-Ausschnitt als **Mono-Mix** (Summe der vier Stems, optional normiert).
Vereinheitlicht die früher in den Notebooks duplizierten Lade-Helfer.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from bwe import config as cfg
from bwe.data.splits import TrackInfo, get_split


def load_mix(
    track: TrackInfo,
    seconds: float = 6.0,
    offset: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Mono-Mix-Ausschnitt (Summe der Stems) eines Tracks als ``float32``-Wellenform."""
    n, start = int(seconds * cfg.SR), int(offset * cfg.SR)
    mix = None
    for stem in cfg.STEMS:
        data, _ = sf.read(str(track.stems[stem]), start=start, frames=n,
                          always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        mix = mono if mix is None else mix + mono
    if normalize:
        mix = mix / max(1e-9, float(np.max(np.abs(mix))))
    return mix.astype(np.float32)


def load_demo(
    split: str = "train",
    index: int = 0,
    seconds: float = 6.0,
    offset: float = 10.0,
    normalize: bool = True,
):
    """Bequeme Variante: Track per ``(split, index)`` wählen. Gibt ``(name, wave)`` zurück."""
    track = get_split(split)[index]
    return track.name, load_mix(track, seconds, offset, normalize)
