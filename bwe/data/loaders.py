"""Audio-Lader für Notebooks/Demos — bewusst getrennt von der tf.data-Pipeline.

Eine Funktion :func:`load_demo` lädt einen Track-Ausschnitt als Mono-Wellenform.
Über ``stems`` wählt man, was überlagert wird: ``"mix"`` (alle vier Stems = der
Mix) oder ein einzelner Stem bzw. eine beliebige Kombination.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from bwe import config as cfg
from bwe.data.splits import get_split


def _resolve_stems(stems) -> tuple[str, ...]:
    """``stems`` -> Tupel gültiger Stem-Namen. ``"mix"`` = alle Stems."""
    if stems == "mix":
        return cfg.STEMS
    sel = (stems,) if isinstance(stems, str) else tuple(stems)
    unknown = [s for s in sel if s not in cfg.STEMS]
    if unknown:
        raise ValueError(f"Unbekannte Stems {unknown}; erlaubt: {cfg.STEMS} oder 'mix'")
    return sel


def load_demo(
    split: str = "train",
    index: int = 0,
    seconds: float | None = 6.0,
    offset: float = 10.0,
    stems="mix",
    normalize: bool = True,
):
    """Lädt einen Mono-Ausschnitt eines Tracks. Gibt ``(name, wave)`` zurück.

    Parameters
    ----------
    split, index : Track per Split (``get_split`` validiert ``split``) und Position.
    seconds, offset : Länge und Startzeit des Ausschnitts in Sekunden. ``seconds=None``
        lädt den **ganzen** Track ab ``offset`` (für die Aggregat-Auswertung). Ragt
        ``offset`` (+ ``seconds``) über das Track-Ende hinaus, wird der Start so weit
        zurückgeschoben, dass noch ``seconds`` Audio geliefert werden (bei kürzeren
        Tracks der ganze Track) — sonst kämen bei kurzen Tracks (z. B. den
        ``Music Delta``-Ausschnitten, ab 13 s) leere Arrays zurück.
    stems : ``"mix"`` (alle Stems überlagert), ein Stem-Name (z. B. ``"drums"``)
        oder eine Kombination (z. B. ``("drums", "bass")``).
    normalize : auf Spitzenpegel 1 normieren.
    """
    track = get_split(split)[index]              # get_split prüft split ∈ SPLIT_NAMES
    sel = _resolve_stems(stems)

    track_len = sf.info(str(track.stems[sel[0]])).frames   # Stems sind gleich lang
    start = int(offset * cfg.SR)
    if seconds is None:
        n = -1                                             # bis Trackende
        start = min(start, max(0, track_len - 1))          # nie hinter dem Ende starten
    else:
        n = int(seconds * cfg.SR)
        start = min(start, max(0, track_len - n))          # Segment ins Track-Ende schieben
    wave = None
    for stem in sel:
        data, _ = sf.read(str(track.stems[stem]), start=start, frames=n,
                          always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        wave = mono if wave is None else wave + mono

    if normalize:
        wave = wave / max(1e-9, float(np.max(np.abs(wave))))
    return track.name, wave.astype(np.float32)
