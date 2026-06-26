"""Sample-Daten bereitstellen — OHNE die 22-GB-ZIP vollständig zu entpacken.

Liest für eine kuratierte Track-Auswahl die Stems **direkt aus der ZIP**
(``zipfile`` → ``BytesIO`` → ``soundfile``), resampelt jeden Stem mit soxr
44.1 → 32 kHz und schreibt nur die 32-kHz-WAVs in den Cache (Default außerhalb
OneDrive, siehe ``bwe.config.DATA_ROOT``). ``mixture`` wird NICHT gespeichert —
es ist die Summe der vier Stems und wird in der Pipeline rekonstruiert.

Auswahl so getroffen, dass alle drei Splits lokal nicht leer sind:
* train (Nicht-Validation) → landen im 'train'-Split
* train (Validation, aus mus.yaml) → landen im 'valid'-Split
* test → 'test'-Split

Idempotent: vorhandene Ausgaben werden übersprungen.

Aufruf:  ../../venv/Scripts/python.exe scripts/prepare_sample.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import librosa
import soundfile as sf

# bwe importierbar machen, auch ohne installiertes Paket
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe import config as cfg  # noqa: E402

# --------------------------------------------------------------------------- #
# Kuratierte Track-Auswahl (exakte Ordnernamen aus der ZIP)
# --------------------------------------------------------------------------- #
SELECTION = {
    "train": [
        # Nicht-Validation → 'train'-Split
        "A Classic Education - NightOwl",
        "ANiMAL - Clinic A",
        "ANiMAL - Easy Tiger",
        "Skelpolu - Together Alone",
        # Validation (in mus.yaml gelistet) → 'valid'-Split
        "ANiMAL - Rockshow",
        "Actions - One Minute Smile",
    ],
    "test": [
        "AM Contra - Heart Peripheral",
        "Al James - Schoolboy Facination",
    ],
}


def _resample_stem(raw_bytes: bytes) -> tuple:
    """WAV-Bytes (44.1 kHz) → (resampled_data [frames, ch], subtype)."""
    with sf.SoundFile(io.BytesIO(raw_bytes)) as f:
        subtype = f.subtype
        sr = f.samplerate
        data = f.read(always_2d=True, dtype="float32")  # [frames, ch]
    y = data.T  # [ch, frames] — Zeit auf letzter Achse für librosa
    yr = librosa.resample(y, orig_sr=sr, target_sr=cfg.SR, res_type="soxr_hq", axis=-1)
    return yr.T, subtype  # [frames, ch]


def main() -> int:
    zip_path = cfg.RAW_ZIP
    if not zip_path.exists():
        print(f"FEHLER: ZIP nicht gefunden: {zip_path}", file=sys.stderr)
        return 1

    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"ZIP:   {zip_path}")
    print(f"Cache: {cfg.DATA_ROOT}\n")

    written = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        members = set(z.namelist())
        for subset, tracks in SELECTION.items():
            for track in tracks:
                out_dir = cfg.DATA_ROOT / subset / track
                out_dir.mkdir(parents=True, exist_ok=True)
                for stem in cfg.STEMS:
                    member = f"{subset}/{track}/{stem}.wav"
                    out_path = out_dir / f"{stem}.wav"
                    if out_path.exists():
                        skipped += 1
                        continue
                    if member not in members:
                        print(f"  !! fehlt in ZIP: {member}", file=sys.stderr)
                        continue
                    data, subtype = _resample_stem(z.read(member))
                    sf.write(out_path, data, cfg.SR, subtype=subtype)
                    written += 1
                    print(f"  + {subset}/{track}/{stem}.wav  ({data.shape[0]} frames, {subtype})")

    print(f"\nFertig: {written} geschrieben, {skipped} übersprungen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
