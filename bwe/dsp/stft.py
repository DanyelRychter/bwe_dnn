"""STFT / iSTFT mit sauberem Nyquist-Handling.

Konventionen
------------
* Spektrogramme sind **frequenz-zuerst**: ``[F, T]`` (F = Bins, T = Frames).
  ``tf.signal`` liefert ``[T, F]`` — wir transponieren intern, damit die Frequenz
  in allen nachgelagerten Modulen (Bandlimit, Copy-Up, Pipeline) auf Achse 0 liegt.
* Volles Spektrogramm hat ``N_BINS_FULL = 513`` Bins. Fürs Netz wird der
  **Nyquist-Bin** (Index 512, 16 kHz, energiearm) weggelassen → 512 Bins, glatt
  durch 16 teilbar. Vor der iSTFT wird er als Null wieder angehängt (die iSTFT
  erwartet zwingend 513 Bins).

Round-Trip ist exakt, wenn die Signallänge ein STFT-Raster genau füllt
(z. B. ``cfg.SEG_SAMPLES``); an den Rändern wirkt die Fensterung.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg


def _inverse_window_fn():
    """Synthese-Fenster, das zum periodischen Hann-Analysefenster die COLA-Bedingung erfüllt."""
    return tf.signal.inverse_stft_window_fn(
        cfg.HOP, forward_window_fn=tf.signal.hann_window
    )


def stft(wave) -> tf.Tensor:
    """Wellenform ``[n]`` → komplexes Spektrogramm ``[F=513, T]`` (complex64)."""
    wave = tf.convert_to_tensor(wave, dtype=tf.float32)
    spec = tf.signal.stft(
        wave,
        frame_length=cfg.N_FFT,
        frame_step=cfg.HOP,
        fft_length=cfg.N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # [T, F]
    return tf.transpose(spec, perm=[1, 0])  # [F, T]


def istft(spec) -> tf.Tensor:
    """Komplexes Spektrogramm ``[F=513, T]`` → Wellenform ``[n]``."""
    spec = tf.convert_to_tensor(spec, dtype=tf.complex64)
    spec_tf = tf.transpose(spec, perm=[1, 0])  # [T, F]
    wave = tf.signal.inverse_stft(
        spec_tf,
        frame_length=cfg.N_FFT,
        frame_step=cfg.HOP,
        fft_length=cfg.N_FFT,
        window_fn=_inverse_window_fn(),
    )
    return wave


def drop_nyquist(spec) -> tf.Tensor:
    """``[513, ...]`` → ``[512, ...]`` (Nyquist-Bin = letzter Frequenz-Bin entfernt)."""
    return spec[: cfg.N_BINS_NET]


def pad_nyquist(spec) -> tf.Tensor:
    """``[512, ...]`` → ``[513, ...]`` (Nyquist-Bin als Null wieder angehängt)."""
    spec = tf.convert_to_tensor(spec)
    zero_shape = tf.concat([[1], tf.shape(spec)[1:]], axis=0)
    zero_bin = tf.zeros(zero_shape, dtype=spec.dtype)
    return tf.concat([spec, zero_bin], axis=0)
