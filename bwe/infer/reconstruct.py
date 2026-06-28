"""Copy-Up-Baseline End-to-End (Stufe 0) — Audio → Rekonstruktion, ohne Lernen.

Kette: ``stft → compress → drop_nyquist → copy_up_hf → decompress →
splice(LF original) → pad_nyquist → istft``. Das LF wird **bit-genau** aus dem
Eingang durchgesplict, nur das HF kommt aus der Hochkopie. Das ist der naive Anker,
gegen den Regression und GAN später gemessen werden.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg
from bwe.dsp.bandlimit import bandlimit
from bwe.dsp.compress import compress, decompress
from bwe.dsp.copyup import copy_up_hf
from bwe.dsp.stft import drop_nyquist, istft, pad_nyquist, stft
from bwe.losses import splice


def reconstruct_copyup(
    input_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    energy_match: bool = True,
) -> tf.Tensor:
    """Bandbegrenzte Eingangs-Wellenform → Vollband-Rekonstruktion (Wellenform).

    Erwartet ein bereits bandbegrenztes Signal (das echte Tiefband). Das HF wird
    per Copy-Up im komprimierten Bereich erzeugt; das LF bleibt unangetastet.
    """
    input_wave = tf.convert_to_tensor(input_wave, tf.float32)
    spec = stft(input_wave)                                   # [513, T] complex

    comp = drop_nyquist(compress(spec))                       # [512, T] komprimiert
    cu = copy_up_hf(comp, cutoff_bin=cutoff_bin, energy_match=energy_match)
    hf = decompress(cu)                                       # [512, T]

    lf = drop_nyquist(spec)[:cutoff_bin]                      # [cutoff, T] LF bit-genau
    spliced = tf.concat([lf, hf[cutoff_bin:]], axis=0)        # [512, T]
    return istft(pad_nyquist(spliced))                        # Vollband-Wellenform


def baseline_from_fullband(
    fullband_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    energy_match: bool = True,
):
    """Komfort für Eval/Demo: Vollband-Signal bandbegrenzen, dann rekonstruieren.

    Returns ``(pred_wave, input_wave)`` — die Rekonstruktion und der bandbegrenzte
    Eingang (z. B. für das Spektrogramm-Tripel).
    """
    input_wave = bandlimit(fullband_wave, cutoff_bin)
    pred = reconstruct_copyup(input_wave, cutoff_bin, energy_match)
    return pred, input_wave


def _model_input(input_wave, cutoff_bin):
    """Bandbegrenzte Wellenform → Modelleingang ``[512, T, 3]`` (komprimiert, Copy-Up-gefüllt)."""
    spec = stft(input_wave)
    cu = copy_up_hf(drop_nyquist(compress(spec)), cutoff_bin=cutoff_bin)       # [512, T] complex
    ri = tf.stack([tf.math.real(cu), tf.math.imag(cu)], axis=-1)              # [512, T, 2]
    freq = tf.cast(tf.linspace(0.0, 1.0, cfg.N_BINS_NET), tf.float32)[:, None, None]
    freq = tf.tile(freq, [1, tf.shape(ri)[1], 1])                            # [512, T, 1]
    return spec, tf.concat([ri, freq], axis=-1)                              # [512, T, 3]


def reconstruct_model(generator, input_wave, cutoff_bin: int = cfg.CUTOFF_BIN) -> tf.Tensor:
    """Vollband-Rekonstruktion mit trainiertem Generator: HF gelernt, LF bit-genau gesplict."""
    input_wave = tf.convert_to_tensor(input_wave, tf.float32)
    spec, inp = _model_input(input_wave, cutoff_bin)
    out = generator(inp[tf.newaxis], training=False)                          # [1, 512, T, 2]
    spliced = splice(out, inp[tf.newaxis], cutoff_bin)[0]                     # LF=inp, HF=Modell
    pred = decompress(tf.complex(spliced[..., 0], spliced[..., 1]))           # [512, T] complex
    lf = drop_nyquist(spec)[:cutoff_bin]                                      # original LF bit-genau
    full = tf.concat([lf, pred[cutoff_bin:]], axis=0)
    return istft(pad_nyquist(full))


def model_from_fullband(generator, fullband_wave, cutoff_bin: int = cfg.CUTOFF_BIN):
    """Wie :func:`baseline_from_fullband`, aber mit dem trainierten Modell."""
    input_wave = bandlimit(fullband_wave, cutoff_bin)
    return reconstruct_model(generator, input_wave, cutoff_bin), input_wave
