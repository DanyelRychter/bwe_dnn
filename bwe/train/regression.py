"""Stufe-1-Training: komplexe Regression (U-Net, RI+Mag-Loss, nur HF).

`RegressionModel` ist eine Keras-`Model`-Subklasse um den Generator: pro Step
`generator → splice → ri_mag_loss`; Monitor-Metrik `lsd_hf` (spektral). So liefert
`fit()` Callbacks (Resume/Checkpoint/EarlyStopping) frei Haus.

Modi (CLI):
  overfit  — ein fester Batch, viele Steps, kein Val → Loss muss ~0 werden (Korrektheit).
  subset   — wenige Tracks (Architektur/Hyperparameter).
  full     — voller Datensatz (final, auf Kaggle).

Aufruf z. B.:  python -m bwe.train.regression --mode overfit --batch 4 --steps 400
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf                                  # noqa: E402
from tensorflow import keras                             # noqa: E402

from bwe import config as cfg                            # noqa: E402
from bwe.data import pipeline as PL                      # noqa: E402
from bwe.eval.metrics import lsd_hf_spec                 # noqa: E402
from bwe.losses import ri_mag_loss, splice              # noqa: E402
from bwe.models.generator import Generator              # noqa: E402
from bwe.train import checkpointing as CK                # noqa: E402


class RegressionModel(keras.Model):
    """U-Net + Splicing + HF-Loss. Eingabe/Ziel: ``(input_spec[..,3], target_spec[..,2])``."""

    def __init__(self, cutoff_bin: int = cfg.CUTOFF_BIN, **kw):
        super().__init__(**kw)
        self.generator = Generator()
        self.cutoff_bin = cutoff_bin
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.lsd_tracker = keras.metrics.Mean(name="lsd_hf")

    @property
    def metrics(self):
        return [self.loss_tracker, self.lsd_tracker]

    def call(self, x, training=False):
        return self.generator(x, training=training)

    def _eval(self, inp, tgt, training):
        out = self.generator(inp, training=training)
        spliced = splice(out, inp, self.cutoff_bin)
        loss = ri_mag_loss(spliced, tgt, self.cutoff_bin)
        return spliced, loss

    def train_step(self, data):
        inp, tgt = data
        with tf.GradientTape() as tape:
            spliced, loss = self._eval(inp, tgt, training=True)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.lsd_tracker.update_state(lsd_hf_spec(spliced, tgt, self.cutoff_bin))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inp, tgt = data
        spliced, loss = self._eval(inp, tgt, training=False)
        self.loss_tracker.update_state(loss)
        self.lsd_tracker.update_state(lsd_hf_spec(spliced, tgt, self.cutoff_bin))
        return {m.name: m.result() for m in self.metrics}


def build_model(lr: float = cfg.LR, beta_1: float = cfg.ADAM_BETA_1,
                cutoff_hz: int = cfg.CUTOFF_HZ) -> RegressionModel:
    model = RegressionModel(cutoff_bin=cfg.cutoff_bin_for(cutoff_hz))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1))
    return model


# --------------------------------------------------------------------------- #
def overfit_one_batch(steps: int = 400, batch_size: int = 4, log_every: int = 25,
                      lr: float = cfg.LR):
    """Korrektheitsbeweis: einen festen Batch auswendig lernen → Loss → ~0."""
    ds = PL.make_dataset("train", batch_size=batch_size, augment=False,
                         shuffle=False, repeat=False)
    batch = next(iter(ds.take(1)))                       # einmal materialisieren = fix
    single = tf.data.Dataset.from_tensors(batch).repeat()
    model = build_model(lr=lr)
    model.fit(single, epochs=max(1, steps // log_every),
              steps_per_epoch=log_every, verbose=2)
    final = float(model.loss_tracker.result())
    print(f"[overfit] final loss = {final:.6f}")
    return model, final


def train(run: str = "reg", mode: str = "full", batch_size: int = cfg.BATCH_SIZE,
          epochs: int = cfg.EPOCHS, limit: int | None = None):
    """Subset-/Voll-Training mit deterministischem Val + Checkpointing (Kaggle)."""
    if limit is None and mode == "subset":
        limit = 20
    train_ds = PL.make_dataset("train", batch_size=batch_size, limit=limit)
    val_ds = PL.make_eval_dataset("valid", batch_size=batch_size, limit=limit)
    spe = PL.steps_per_epoch_for("train", batch_size, limit=limit)
    model = build_model()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     steps_per_epoch=spe, callbacks=CK.callbacks(run), verbose=2)
    # Generator-Gewichte separat sichern (Warm-Start für Phase D / Inferenz).
    model.generator.save_weights(str(CK.run_dir(run) / "generator.weights.h5"))
    return model, hist


def main():
    ap = argparse.ArgumentParser(description="BWE Stufe-1-Regression")
    ap.add_argument("--mode", choices=["overfit", "subset", "full"], default="overfit")
    ap.add_argument("--run", default="reg")
    ap.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--steps", type=int, default=400, help="Overfit: Gesamt-Steps")
    ap.add_argument("--lr", type=float, default=cfg.LR, help="Lernrate")
    ap.add_argument("--limit", type=int, default=None, help="nur N Tracks")
    args = ap.parse_args()

    print(cfg.summary())
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus or 'keine (CPU)'}")

    if args.mode == "overfit":
        overfit_one_batch(steps=args.steps, batch_size=args.batch, lr=args.lr)
    else:
        train(run=args.run, mode=args.mode, batch_size=args.batch,
              epochs=args.epochs, limit=args.limit)


if __name__ == "__main__":
    main()
