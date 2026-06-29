"""Checkpointing/Resume-Callbacks — robust gegen Abbrüche (12-h-Grenze auf Kaggle).

Schreibt bewusst nach ``cfg.CKPT_ROOT`` (außerhalb von OneDrive; OneDrive sperrt
Backup-/CSV-Dateien).

* **Regression** (:func:`callbacks`): ``BackupAndRestore`` + ``ModelCheckpoint`` +
  ``CSVLogger`` + ``EarlyStopping`` — das Standard-Keras-Trio.
* **GAN** (:func:`gan_checkpoint`): eigener ``tf.train.Checkpoint`` über *zwei* Optimizer
  (G und D) + Epoche + Best-Wert. ``BackupAndRestore`` würde den D-Optimizer-State über
  Sessions nicht zuverlässig wiederherstellen; der explizite Checkpoint schon. Kein
  ``EarlyStopping`` (``g_loss`` ist kein Qualitätsmaß — Auswahl per ``val_lsd_hf``).
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
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


class GANCheckpoint(keras.callbacks.Callback):
    """Eigener Checkpoint für das GAN: sichert/restored G, D, **beide** Optimizer + Epoche.

    Beim Konstruieren (vor ``fit``) wird der letzte Checkpoint — falls vorhanden —
    wiederhergestellt (Variablen-Restore ist *deferred*: Optimizer-Slots werden beim
    ersten Step gefüllt). ``initial_epoch`` sagt ``fit``, wo es weitergeht.

    Am Epochenende: Checkpoint speichern, ``generator.weights.h5`` (letzter Stand für
    Inferenz/Notebook) und bei neuem Best-``val_lsd_hf`` zusätzlich
    ``best_generator.weights.h5``.
    """

    def __init__(self, run_name: str, model: keras.Model, monitor: str = "val_lsd_hf",
                 max_to_keep: int = 3):
        super().__init__()
        self.dir = run_dir(run_name)
        self.monitor = monitor
        self.epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.best_var = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)
        self.ckpt = tf.train.Checkpoint(
            generator=model.generator,
            discriminator=model.discriminator,
            g_optimizer=model.g_optimizer,
            d_optimizer=model.d_optimizer,
            epoch=self.epoch_var,
            best=self.best_var,
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt, str(self.dir / "ckpt"), max_to_keep=max_to_keep
        )
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print(f"[gan-ckpt] fortgesetzt ab Epoche {int(self.epoch_var.numpy())} "
                  f"({self.manager.latest_checkpoint})")

    @property
    def initial_epoch(self) -> int:
        return int(self.epoch_var.numpy())

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_var.assign(epoch + 1)
        self.manager.save()
        gen = self.model.generator
        gen.save_weights(str(self.dir / "generator.weights.h5"))
        current = logs.get(self.monitor)
        if current is not None and current < float(self.best_var.numpy()):
            self.best_var.assign(float(current))
            gen.save_weights(str(self.dir / "best_generator.weights.h5"))


def gan_checkpoint(run_name: str, model: keras.Model,
                   monitor: str = "val_lsd_hf") -> GANCheckpoint:
    """Convenience-Factory (analog zu :func:`callbacks`)."""
    return GANCheckpoint(run_name, model, monitor=monitor)
