"""Schritt 13 — PatchGAN-Diskriminator (Shapes, Feature-Liste, Spectral Norm)."""

import tensorflow as tf
from tensorflow import keras

from bwe import config as cfg
from bwe.models.discriminator import Discriminator, build_discriminator


def test_forward_returns_logits_and_feats():
    """Eingabe ``[B,512,T,2]`` → Patch-Logits (1 Kanal) + eine Aktivierung je Conv-Block."""
    d = build_discriminator()
    x = tf.zeros([2, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_OUTPUT_CHANNELS])
    logits, *feats = d(x)
    assert logits.shape[0] == 2 and logits.shape[-1] == 1
    assert len(feats) == len(cfg.DISC_CHANNELS)
    assert feats[0].shape[-1] == cfg.DISC_CHANNELS[0]


def test_patch_grid_downsampled():
    """3× Stride-2 → Patch-Gitter deutlich kleiner als der Eingang (lokales Urteil)."""
    d = build_discriminator()
    logits = d(tf.zeros([1, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_OUTPUT_CHANNELS]))[0]
    assert logits.shape[1] < cfg.N_BINS_NET
    assert logits.shape[2] < cfg.SEG_FRAMES


def test_odd_time_ok():
    """Voll-faltend → krummes T ohne Fehler (keine Teilbarkeits-Constraints, keine Skips)."""
    d = build_discriminator()
    logits = d(tf.zeros([2, cfg.N_BINS_NET, 125, cfg.N_OUTPUT_CHANNELS]))[0]
    assert logits.shape[0] == 2


def test_spectral_norm_present():
    """Spectral Norm aktiv → mindestens eine SpectralNormalization-Schicht."""
    d = build_discriminator()
    assert any(isinstance(layer, keras.layers.SpectralNormalization)
               for layer in d.layers)


def test_summary_runs():
    Discriminator().summary()
