"""Kanonischer MUSDB18-Split (86 / 14 / 50) — eigenständig, ohne ffmpeg.

Ursprünglich war ``musdb`` als Split-Quelle geplant. ``import musdb`` zieht aber
``stempeg`` nach, das beim Import zwingend **ffmpeg/ffprobe** verlangt — für unsere
WAV-Stems völlig unnötig und genau das, was das lokale Debugging (ohne ffmpeg)
bräche. Die Split-Regel selbst ist trivial, deshalb replizieren wir sie hier 1:1:

* physische Ordner ``train/`` und ``test/`` durchlaufen (alphabetisch sortiert,
  wie ``musdb``),
* die 14 vorgegebenen Validation-Tracks (aus ``musdb/configs/mus.yaml``) aus dem
  train-Ordner herausschneiden → 'valid', der Rest → 'train'.

Ergebnis ist identisch zu ``musdb`` — aber ohne Fremdabhängigkeit, lokal wie auf
Kaggle. Ein zusätzlicher Kaggle-``val/``-Ordner wird ignoriert (wir laufen nur über
``train/`` und ``test/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bwe import config as cfg

VALID_SPLITS = ("train", "valid", "test")

# Die kanonischen 14 Validation-Tracks (Quelle: musdb/configs/mus.yaml).
VALIDATION_TRACKS = frozenset({
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
})


@dataclass(frozen=True)
class TrackInfo:
    name: str
    subset: str                 # 'train' oder 'test' (physischer Ordner)
    stems: dict                 # {'vocals': Path, 'drums': Path, ...}

    def stem_path(self, name: str) -> Path:
        return self.stems[name]


def _collect(track_dir: Path) -> dict:
    """{stem: Path} für alle vorhandenen Stems eines Track-Ordners."""
    return {
        stem: track_dir / f"{stem}.wav"
        for stem in cfg.STEMS
        if (track_dir / f"{stem}.wav").exists()
    }


def get_split(split: str, root: Path | None = None) -> list[TrackInfo]:
    """Tracks eines Splits ∈ {'train','valid','test'} als Liste von :class:`TrackInfo`."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split muss aus {VALID_SPLITS} sein, war {split!r}")

    data_root = Path(root) if root is not None else cfg.DATA_ROOT
    subset = "test" if split == "test" else "train"
    subset_dir = data_root / subset
    if not subset_dir.is_dir():
        return []

    tracks = []
    for track_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
        if split == "train" and track_dir.name in VALIDATION_TRACKS:
            continue
        if split == "valid" and track_dir.name not in VALIDATION_TRACKS:
            continue
        stems = _collect(track_dir)
        if all(s in stems for s in cfg.STEMS):       # nur vollständige Tracks
            tracks.append(TrackInfo(name=track_dir.name, subset=subset, stems=stems))
    return tracks


def all_splits(root: Path | None = None) -> dict:
    """{'train': [...], 'valid': [...], 'test': [...]} — bequem für Checks/Reports."""
    return {s: get_split(s, root=root) for s in VALID_SPLITS}


if __name__ == "__main__":
    for split, tracks in all_splits().items():
        print(f"{split:6s}: {len(tracks):3d} Tracks")
        for t in tracks:
            print(f"         - {t.name}")
