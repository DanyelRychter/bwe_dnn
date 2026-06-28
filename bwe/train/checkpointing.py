"""Checkpointing/Resume-Callbacks — robust gegen Abbrüche (12-h-Grenze auf Kaggle).

Schreibt bewusst nach ``cfg.CKPT_ROOT`` (außerhalb von OneDrive; OneDrive sperrt
Backup-/CSV-Dateien). Das Callback-Trio:
* ``BackupAndRestore`` — Resume nach Absturz (Zelle einfach erneut ausführen).
* ``ModelCheckpoint(save_best_only)`` — bestes Val-Modell als Gewichtsdatei.
* ``CSVLogger(append=True)`` — Metriken je Epoche (Lernkurven überleben Abbruch).
* ``EarlyStopping(restore_best_weights)`` — stoppt bei Stagnation.
"""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras

from bwe import config as cfg


def run_dir(run_name: str) -> Path:
    d = cfg.CKPT_ROOT / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def callbacks(run_name: str, monitor: str = "val_lsd_hf",
              patience: int = cfg.EARLY_STOP_PATIENCE) -> list:
    d = run_dir(run_name)
    return [
        keras.callbacks.BackupAndRestore(str(d / "backup")),
        keras.callbacks.ModelCheckpoint(
            str(d / "best.weights.h5"), monitor=monitor, mode="min",
            save_best_only=True, save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(d / "log.csv"), append=True),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="min", patience=patience, restore_best_weights=True,
        ),
    ]
