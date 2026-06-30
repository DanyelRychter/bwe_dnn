"""Stufe-2-Training: adversariales Feintuning (GAN = U-Net + PatchGAN).

`GANModel` ist eine Keras-`Model`-Subklasse, die Generator *und* Diskriminator hält.
Pro Step zwei `GradientTape`-Blöcke (pix2pix-Muster): der Diskriminator lernt echtes
gegen generiertes HF zu trennen, der Generator lernt, ihn zu täuschen — **zusätzlich**
zum Rekonstruktions-Anker `ri_mag_loss` und dem Feature-Matching. Splicing (echtes LF,
generiertes HF) passiert **vor** dem Diskriminator, damit der adversariale Druck aufs HF
zielt.

Ablauf (Schritt 15):
  1. Generator-**Warm-Start** aus der Regression (`generator.weights.h5`).
  2. **Diskriminator-Vorlauf** (einige hundert Steps, G eingefroren) — gibt dem D einen
     Vorsprung, bevor sein Gradient auf den G wirkt.
  3. Adversariale Phase via `fit()` mit eigenem `GANCheckpoint` (Resume über zwei Optimizer).

`g_loss` ist **kein** Qualitätsmaß (D ist bewegliches Ziel) → Modellwahl per Val-`lsd_hf`
+ Reinhören (Notebook).

Modi (CLI):
  overfit  — ein fester Batch, viele Steps → Divergenz-Check (g/d pendeln, kein Kollaps).
  subset   — wenige Tracks (λ-Werte/Lernraten festklopfen).
  full     — voller Datensatz (final, auf Kaggle).

Aufruf z. B.:  python -m bwe.train.gan --mode subset --warm-start <pfad>/generator.weights.h5
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf                                  # noqa: E402
from tensorflow import keras                             # noqa: E402

from bwe import config as cfg                            # noqa: E402
from bwe.data import pipeline as PL                      # noqa: E402
from bwe.eval.metrics import lsd_hf_spec                 # noqa: E402
from bwe.losses import (                                 # noqa: E402
    discriminator_loss, feature_matching_loss, generator_adv_loss,
    ri_mag_loss, splice,
)
from bwe.models.discriminator import Discriminator      # noqa: E402
from bwe.models.generator import Generator              # noqa: E402
from bwe.train import checkpointing as CK                # noqa: E402


class GANModel(keras.Model):
    """U-Net (G) + PatchGAN (D), zweiphasiges adversariales Training.

    Eingabe/Ziel je Batch: ``(input_spec[..,3], target_spec[..,2])`` — Target ist das
    echte Vollband (= „real" für den Diskriminator).
    """

    def __init__(self, cutoff_bin: int = cfg.CUTOFF_BIN,
                 lambda_adv: float = cfg.LAMBDA_ADV,
                 lambda_fm: float = cfg.LAMBDA_FM, **kw):
        super().__init__(**kw)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.cutoff_bin = cutoff_bin
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.adv_tracker = keras.metrics.Mean(name="adv")
        self.fm_tracker = keras.metrics.Mean(name="fm")
        self.lsd_tracker = keras.metrics.Mean(name="lsd_hf")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.recon_tracker,
                self.adv_tracker, self.fm_tracker, self.lsd_tracker]

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def call(self, x, training=False):
        return self.generator(x, training=training)

    # --- Training ---------------------------------------------------------- #
    def train_step(self, data):
        inp, tgt = data
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            gen_out = self.generator(inp, training=True)
            fake = splice(gen_out, inp, self.cutoff_bin)          # echtes LF + generiertes HF
            real_logits, *real_feats = self.discriminator(tgt, training=True)
            fake_logits, *fake_feats = self.discriminator(fake, training=True)

            d_loss = discriminator_loss(real_logits, fake_logits)

            recon = ri_mag_loss(fake, tgt, self.cutoff_bin)
            adv = generator_adv_loss(fake_logits)
            fm = feature_matching_loss(real_feats, fake_feats)    # real_feats ⟂ G → const
            g_loss = recon + self.lambda_adv * adv + self.lambda_fm * fm

        d_vars = self.discriminator.trainable_variables
        g_vars = self.generator.trainable_variables
        self.d_optimizer.apply_gradients(zip(d_tape.gradient(d_loss, d_vars), d_vars))
        self.g_optimizer.apply_gradients(zip(g_tape.gradient(g_loss, g_vars), g_vars))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.recon_tracker.update_state(recon)
        self.adv_tracker.update_state(adv)
        self.fm_tracker.update_state(fm)
        self.lsd_tracker.update_state(lsd_hf_spec(fake, tgt, self.cutoff_bin))
        return {m.name: m.result() for m in self.metrics}

    def d_train_step(self, data):
        """Nur-Diskriminator-Step (Vorlauf, Generator eingefroren)."""
        inp, tgt = data
        fake = splice(self.generator(inp, training=False), inp, self.cutoff_bin)
        with tf.GradientTape() as d_tape:
            real_logits = self.discriminator(tgt, training=True)[0]
            fake_logits = self.discriminator(fake, training=True)[0]
            d_loss = discriminator_loss(real_logits, fake_logits)
        d_vars = self.discriminator.trainable_variables
        self.d_optimizer.apply_gradients(zip(d_tape.gradient(d_loss, d_vars), d_vars))
        return d_loss

    def test_step(self, data):
        """Val: nur Generator-Qualität (``lsd_hf`` + ``recon``); ``g_loss`` ist kein Maß."""
        inp, tgt = data
        fake = splice(self.generator(inp, training=False), inp, self.cutoff_bin)
        self.recon_tracker.update_state(ri_mag_loss(fake, tgt, self.cutoff_bin))
        self.lsd_tracker.update_state(lsd_hf_spec(fake, tgt, self.cutoff_bin))
        return {m.name: m.result() for m in self.metrics}


def build_model(warm_start: str | None = None, lr: float = cfg.GAN_LR,
                beta_1: float = cfg.GAN_BETA_1, cutoff_hz: int = cfg.CUTOFF_HZ,
                lambda_adv: float = cfg.LAMBDA_ADV,
                lambda_fm: float = cfg.LAMBDA_FM) -> GANModel:
    model = GANModel(cutoff_bin=cfg.cutoff_bin_for(cutoff_hz),
                     lambda_adv=lambda_adv, lambda_fm=lambda_fm)
    model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
        g_optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
    )
    # Variablen anlegen (Dummy-Forwards) — Checkpoint/Restore verlangen gebaute Modelle.
    # Diskriminator zuerst, damit beim Markieren von ``model`` kein Sub-Layer „unbuilt" ist.
    model.discriminator(
        tf.zeros((1, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_OUTPUT_CHANNELS))
    )
    model(tf.zeros((1, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_INPUT_CHANNELS)))
    if warm_start:
        model.generator.load_weights(warm_start)
        print(f"[gan] Generator warm-gestartet aus {warm_start}")
    return model


def pretrain_discriminator(model: GANModel, ds, steps: int, log_every: int = 50):
    """Diskriminator-Vorlauf: ``steps`` Nur-D-Updates (G eingefroren)."""
    step_fn = tf.function(model.d_train_step)
    it = iter(ds)
    for i in range(steps):
        dl = step_fn(next(it))
        if (i + 1) % log_every == 0:
            print(f"[d-warmup] {i + 1}/{steps}  d_loss={float(dl):.4f}")


# --------------------------------------------------------------------------- #
def overfit_one_batch(steps: int = 400, batch_size: int = 4, log_every: int = 25,
                      warm_start: str | None = None):
    """Divergenz-Check: einen festen Batch adversarial trainieren (kein Kollaps erwartet)."""
    ds = PL.make_dataset("train", batch_size=batch_size, augment=False,
                         shuffle=False, repeat=False)
    batch = next(iter(ds.take(1)))
    single = tf.data.Dataset.from_tensors(batch).repeat()
    model = build_model(warm_start=warm_start)
    model.fit(single, epochs=max(1, steps // log_every),
              steps_per_epoch=log_every, verbose=2)
    print(f"[overfit] d_loss={float(model.d_loss_tracker.result()):.4f} "
          f"g_loss={float(model.g_loss_tracker.result()):.4f} "
          f"lsd_hf={float(model.lsd_tracker.result()):.4f}")
    return model


def train(run: str = "gan", mode: str = "full", warm_start: str | None = None,
          batch_size: int = cfg.GAN_BATCH_SIZE, epochs: int = cfg.GAN_EPOCHS,
          limit: int | None = None, d_warmup: int = cfg.GAN_D_WARMUP_STEPS):
    """Adversariales Feintuning mit D-Vorlauf, deterministischem Val + GAN-Checkpointing."""
    if limit is None and mode == "subset":
        limit = 20
    train_ds = PL.make_dataset("train", batch_size=batch_size, limit=limit)
    val_ds = PL.make_eval_dataset("valid", batch_size=batch_size, limit=limit)
    spe = PL.steps_per_epoch_for("train", batch_size, limit=limit)

    model = build_model(warm_start=warm_start)
    ckpt = CK.gan_checkpoint(run, model)
    # D-Vorlauf nur beim frischen Start (beim Resume hat der D ihn schon hinter sich).
    if ckpt.initial_epoch == 0 and d_warmup > 0:
        pretrain_discriminator(model, train_ds, d_warmup)

    csv = keras.callbacks.CSVLogger(str(CK.run_dir(run) / "log.csv"), append=True)
    cbs = [ckpt, csv]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     initial_epoch=ckpt.initial_epoch, steps_per_epoch=spe,
                     callbacks=cbs, verbose=2)
    CK.log_stop_reason(hist, cbs, epochs)            # Abschlussgrund explizit ins Log
    # End-Gewichte sichern (G für Inferenz/Notebook, D für etwaiges Curriculum-Warm-Start).
    model.generator.save_weights(str(CK.run_dir(run) / "generator.weights.h5"))
    model.discriminator.save_weights(str(CK.run_dir(run) / "discriminator.weights.h5"))
    return model, hist


def train_resumable(run: str = "gan_full", mode: str = "full", warm_start=None,
                    ckpt_src=None, **kw):
    """Cold-Start, Generator-Warm-Start oder exaktes Resume — je nach ``ckpt_src``.

    Macht das Voll-Notebook im **Commit-Modus** out-of-the-box (top-to-bottom, ohne einzelne
    Zellen erneut auszuführen). Anders als bei der Regression liegt der exakte GAN-Zustand
    im eigenen ``tf.train.Checkpoint`` unter ``ckpt/`` (G, D, beide Optimizer, Epoche).

    * ``ckpt_src=None`` → **Cold-Start**; ``warm_start`` (Phase-C-Generator) wie übergeben.
    * ``ckpt_src`` enthält ``ckpt/`` → **exaktes Resume**; der ganze Run-Ordner wird nach
      ``cfg.CKPT_ROOT`` gespiegelt, ``GANCheckpoint`` setzt automatisch fort (D-Vorlauf
      entfällt). ``warm_start`` wird ignoriert (der G-Zustand kommt aus ``ckpt/``).
    * ``ckpt_src`` nur mit ``(best_)generator.weights.h5`` → **Generator-Warm-Start** daraus.

    Das ``ckpt/`` stammt typischerweise aus dem **Output einer vorigen Commit-Version**
    (Kaggle sichert ``/kaggle/working`` automatisch) — als Input anhängen und ``ckpt_src``
    darauf zeigen.
    """
    rd = CK.run_dir(run)
    if ckpt_src is None:
        print("[resume] kein ckpt_src -> Cold-Start.")
        return train(run=run, mode=mode, warm_start=warm_start, **kw)

    ckpt_src = Path(ckpt_src)
    if (ckpt_src / "ckpt").exists():
        print(f"[resume] ckpt/ in {ckpt_src} gefunden -> exaktes Resume.")
        for item in ckpt_src.iterdir():
            dst = rd / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy(item, dst)
        return train(run=run, mode=mode, warm_start=None, **kw)  # Zustand kommt aus ckpt/

    gen = ckpt_src / "best_generator.weights.h5"
    if not gen.exists():
        gen = ckpt_src / "generator.weights.h5"
    if gen.exists():
        print(f"[resume] kein ckpt/, aber {gen.name} in {ckpt_src} -> Generator-Warm-Start.")
        return train(run=run, mode=mode, warm_start=str(gen), **kw)

    print(f"[resume] WARNUNG: {ckpt_src} ohne ckpt/ und generator.weights.h5 -> Cold-Start.")
    return train(run=run, mode=mode, warm_start=warm_start, **kw)


def main():
    ap = argparse.ArgumentParser(description="BWE Stufe-2-GAN")
    ap.add_argument("--mode", choices=["overfit", "subset", "full"], default="overfit")
    ap.add_argument("--run", default="gan")
    ap.add_argument("--warm-start", default=None, help="Generator-Warm-Start (.weights.h5)")
    ap.add_argument("--batch", type=int, default=cfg.GAN_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=cfg.GAN_EPOCHS)
    ap.add_argument("--steps", type=int, default=400, help="Overfit: Gesamt-Steps")
    ap.add_argument("--limit", type=int, default=None, help="nur N Tracks")
    ap.add_argument("--d-warmup", type=int, default=cfg.GAN_D_WARMUP_STEPS)
    ap.add_argument("--lambda-adv", type=float, default=cfg.LAMBDA_ADV)
    ap.add_argument("--lambda-fm", type=float, default=cfg.LAMBDA_FM)
    args = ap.parse_args()

    print(cfg.summary())
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus or 'keine (CPU)'}")

    if args.mode == "overfit":
        overfit_one_batch(steps=args.steps, batch_size=args.batch,
                          warm_start=args.warm_start)
    else:
        train(run=args.run, mode=args.mode, warm_start=args.warm_start,
              batch_size=args.batch, epochs=args.epochs, limit=args.limit,
              d_warmup=args.d_warmup)


if __name__ == "__main__":
    main()
