"""PatchGAN-Diskriminator (Stufe 2 — GAN).

Beurteilt das **volle** Spektrogramm ``[B, 512, T, 2]`` (Re/Im) lokal: ein Gitter aus
überlappenden Patches statt eines globalen Urteils. Weil das LF vor dem Diskriminator
per :func:`bwe.losses.splice` aus dem Original übernommen wird (in *echt* wie *fake*
identisch), kann der Diskriminator nur am HF unterscheiden — genau dorthin lenkt das
den adversarialen Druck.

Architektur (Leitfaden §10 / Schritt 13): 3× ``Conv2D`` Stride-2 (64→128→256) mit
``LeakyReLU(0.2)`` und **Spectral Normalization** (stabilisiert das D-Training; ersetzt
BatchNorm, das im Diskriminator adversarial zickt), dann eine finale 1-Kanal-Conv →
Patch-Logits (**linear**, BCE rechnet ``from_logits``).

Voll-faltend → beliebiges T, keine Teilbarkeits-Constraints (keine Skip-Connections).
Der Builder gibt ``[logits, *feats]`` zurück: die Zwischenaktivierungen ``feats`` speisen
den Feature-Matching-Loss (:func:`bwe.losses.feature_matching_loss`).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from bwe import config as cfg

layers = keras.layers


def _sn(layer: keras.layers.Layer) -> keras.layers.Layer:
    """Wickelt eine Schicht in Spectral Normalization (falls aktiviert)."""
    if cfg.DISC_USE_SPECTRAL_NORM:
        return layers.SpectralNormalization(layer)
    return layer


def build_discriminator(
    n_in: int = cfg.N_OUTPUT_CHANNELS,
    channels: tuple[int, ...] = cfg.DISC_CHANNELS,
) -> keras.Model:
    """Funktionaler PatchGAN. Eingabe ``[B, 512, T, 2]`` → ``[logits, *feats]``.

    ``logits``: Patch-Gitter ``[B, F', T', 1]`` (linear). ``feats``: Liste der
    LeakyReLU-Aktivierungen je Conv-Block (für Feature-Matching).
    """
    inp = keras.Input(shape=(cfg.N_BINS_NET, None, n_in), name="spec")

    x = inp
    feats = []
    for i, c in enumerate(channels):
        x = _sn(layers.Conv2D(c, 4, strides=2, padding="same", name=f"disc{i}"))(x)
        x = layers.LeakyReLU(cfg.LEAKY_SLOPE)(x)
        feats.append(x)

    logits = _sn(
        layers.Conv2D(1, 4, strides=1, padding="same", name="patch")
    )(x)
    return keras.Model(inp, [logits, *feats], name="patch_discriminator")


class Discriminator(keras.Model):
    """Dünner Wrapper um :func:`build_discriminator` (analog zu :class:`Generator`)."""

    def __init__(self, n_in: int = cfg.N_OUTPUT_CHANNELS,
                 channels: tuple[int, ...] = cfg.DISC_CHANNELS, **kw):
        super().__init__(name="discriminator", **kw)
        self.net = build_discriminator(n_in, channels)

    def call(self, x, training=False):
        return self.net(x, training=training)

    def summary(self, **kw):
        return self.net.summary(**kw)
