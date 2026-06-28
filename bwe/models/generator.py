"""2D-U-Net-Generator (Regressor = Generator für Stufe 1 und 2).

Voll-faltend: keine Dense-Layer mit fixer Zeitdimension → akzeptiert beliebig lange
Eingaben. Eingang ``[B, 512, T, N_INPUT_CHANNELS]`` (Re/Im/Freq-Koord), Ausgang
``[B, 512, T, N_OUTPUT_CHANNELS]`` (Re/Im, **lineare** Aktivierung — komprimierte
Re/Im sind unbeschränkt und vorzeichenbehaftet).

Stride-2 halbiert pro Ebene **beide** Achsen → F und T müssen Vielfache von
``2**Ebenen = 16`` sein. F=512 ist glatt; die Zeit T wird in :class:`Generator`
dynamisch auf das nächste Vielfache von 16 gepaddet und die Ausgabe zurückgecroppt
(deckt Training mit fester Länge *und* Inferenz beliebiger Länge ab).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from bwe import config as cfg

layers = keras.layers


def _down(filters: int, name: str) -> keras.Sequential:
    """Encoder-Block: Conv2D Stride-2 → BatchNorm → LeakyReLU (halbiert F und T)."""
    return keras.Sequential([
        layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(cfg.LEAKY_SLOPE),
    ], name=name)


def _up(filters: int, name: str) -> keras.Sequential:
    """Decoder-Block: Conv2DTranspose Stride-2 → BatchNorm → LeakyReLU (verdoppelt F und T)."""
    return keras.Sequential([
        layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(cfg.LEAKY_SLOPE),
    ], name=name)


def build_unet(
    n_in: int = cfg.N_INPUT_CHANNELS,
    n_out: int = cfg.N_OUTPUT_CHANNELS,
    channels: tuple[int, ...] = cfg.UNET_CHANNELS,
) -> keras.Model:
    """Funktionaler U-Net-Kern (erwartet T als Vielfaches von ``len(channels)**2``)."""
    inp = keras.Input(shape=(cfg.N_BINS_NET, None, n_in), name="input_spec")

    downs = [_down(c, f"down{i}") for i, c in enumerate(channels)]
    ups = [_up(c, f"up{i}") for i, c in enumerate(reversed(channels[:-1]))]
    last = layers.Conv2DTranspose(
        n_out, 4, strides=2, padding="same", activation="linear", name="out"
    )

    x = inp
    skips = []
    for d in downs:
        x = d(x)
        skips.append(x)
    skips = list(reversed(skips[:-1]))            # Bottleneck (letzter) hat keinen Skip

    for u, skip in zip(ups, skips):
        x = u(x)
        x = layers.Concatenate()([x, skip])       # Skip-Connection (LF-Kontext → Decoder)
    x = last(x)
    return keras.Model(inp, x, name="bwe_unet")


class Generator(keras.Model):
    """U-Net mit dynamischem Zeit-Pad/Crop auf ein Vielfaches von ``UNET_DEPTH_FACTOR``."""

    def __init__(self, n_in: int = cfg.N_INPUT_CHANNELS,
                 n_out: int = cfg.N_OUTPUT_CHANNELS,
                 channels: tuple[int, ...] = cfg.UNET_CHANNELS, **kw):
        super().__init__(name="generator", **kw)
        self.net = build_unet(n_in, n_out, channels)

    def call(self, x, training=False):
        t = tf.shape(x)[2]
        pad = tf.math.floormod(-t, cfg.UNET_DEPTH_FACTOR)     # bis zum nächsten Vielfachen von 16
        x = tf.pad(x, [[0, 0], [0, 0], [0, pad], [0, 0]])
        y = self.net(x, training=training)
        return y[:, :, :t, :]                                  # zurück auf Original-T croppen

    def summary(self, **kw):
        return self.net.summary(**kw)
