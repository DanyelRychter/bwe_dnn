"""Schritt 9 — U-Net-Generator (Shapes, Zeit-Pad/Crop, lineare Ausgabe)."""

import tensorflow as tf

from bwe import config as cfg
from bwe.models.generator import Generator, build_unet


def test_forward_shape():
    g = Generator()
    x = tf.zeros([2, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_INPUT_CHANNELS])
    y = g(x)
    assert tuple(y.shape) == (2, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_OUTPUT_CHANNELS)


def test_odd_time_is_cropped_back():
    """Krummes T (nicht durch 16 teilbar) → intern gepaddet, Ausgabe zurück auf T."""
    g = Generator()
    t = 125
    x = tf.zeros([2, cfg.N_BINS_NET, t, cfg.N_INPUT_CHANNELS])
    y = g(x)
    assert tuple(y.shape) == (2, cfg.N_BINS_NET, t, cfg.N_OUTPUT_CHANNELS)


def test_output_linear_signed():
    """Lineare Ausgabe → Werte beider Vorzeichen möglich (kein ReLU/tanh-Clip)."""
    g = Generator()
    x = tf.random.normal([1, cfg.N_BINS_NET, cfg.SEG_FRAMES, cfg.N_INPUT_CHANNELS])
    y = g(x).numpy()
    assert (y < 0).any() and (y > 0).any()


def test_summary_runs():
    build_unet().summary()
