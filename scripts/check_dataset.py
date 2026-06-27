"""Schritt 0 — Datensatz-Verifikation auf dem 32-kHz-Cache.

Prüft Struktur und Integrität, OHNE TensorFlow zu laden (schnell):
* Anzahl Tracks je Split (train/valid/test) — lokal das Subset, auf Kaggle 86/14/50.
* je Track: alle vier Stems vorhanden.
* je Stem: Samplerate == 32 kHz, Bittiefe (soundfile.info), Dauer.
* Disjunktheit train ∩ test (physische Ordner) und train/valid (kanonischer Split).

Reportet tatsächliche Zahlen und meldet einen Fehler-Exitcode nur bei echten
Struktur-/Integritätsproblemen (fehlende Stems, falsche Samplerate, Überlappung).

Aufruf:  ../../venv/Scripts/python.exe scripts/check_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe import config as cfg              # noqa: E402
from bwe.data import splits as SP          # noqa: E402

EXPECTED_FULL = {"train": 86, "valid": 14, "test": 50}


def main() -> int:
    print(f"DATA_ROOT: {cfg.DATA_ROOT}\n")
    if not cfg.DATA_ROOT.exists():
        print("FEHLER: DATA_ROOT existiert nicht — erst scripts/prepare_sample.py laufen lassen.")
        return 1

    all_tracks = SP.all_splits()
    problems: list[str] = []

    for split in SP.SPLIT_NAMES:
        tracks = all_tracks[split]
        print(f"[{split}]  {len(tracks)} Tracks  (Vollsatz erwartet: {EXPECTED_FULL[split]})")
        for t in tracks:
            # Stem-Vollständigkeit
            missing = [s for s in cfg.STEMS if s not in t.stems]
            if missing:
                problems.append(f"{split}/{t.name}: fehlende Stems {missing}")
            # je Stem: Samplerate, Bittiefe, Dauer
            info = None
            for stem, path in t.stems.items():
                try:
                    info = sf.info(str(path))
                except Exception as e:                       # noqa: BLE001
                    problems.append(f"{split}/{t.name}/{stem}: nicht lesbar ({e})")
                    continue
                if info.samplerate != cfg.SR:
                    problems.append(
                        f"{split}/{t.name}/{stem}: SR={info.samplerate} != {cfg.SR}"
                    )
            if info is not None:
                dur = info.frames / info.samplerate
                print(f"      - {t.name:42s} {dur:5.1f}s  {info.subtype}  {info.channels}ch")
        print()

    # Disjunktheit
    names = {s: {t.name for t in all_tracks[s]} for s in SP.SPLIT_NAMES}
    if names["train"] & names["test"]:
        problems.append(f"train∩test nicht leer: {names['train'] & names['test']}")
    if names["train"] & names["valid"]:
        problems.append(f"train∩valid nicht leer: {names['train'] & names['valid']}")

    print("=" * 60)
    if problems:
        print(f"PROBLEME ({len(problems)}):")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("OK — Struktur, Samplerate (32 kHz) und Split-Disjunktheit in Ordnung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
