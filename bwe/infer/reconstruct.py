"""Copy-Up-Baseline End-to-End (Stufe 0) — Audio → Rekonstruktion, ohne Lernen.

Kette: ``stft → compress → drop_nyquist → copy_up_hf → decompress →
splice(LF original) → pad_nyquist → istft``. Das LF wird **bit-genau** aus dem
Eingang durchgesplict, nur das HF kommt aus der Hochkopie. Das ist der naive Anker,
gegen den Regression und GAN später gemessen werden.

Beide Rekonstruktionen (Copy-Up und Modell) sind voll-faltend und arbeiten daher
auf **beliebig langen** Signalen (ein ganzer Track in einem Durchgang). Mit
``center=True`` (Default) wird das Signal vor der STFT um ``N_FFT`` genullt und der
iSTFT-Output wieder exakt auf die Eingangslänge zugeschnitten (= ``librosa
center=True``) → randsaubere Overlap-Add, keine Block-Klicks am Track-Anfang/-Ende
und ``len(pred) == len(input)``.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg
from bwe.dsp.bandlimit import bandlimit
from bwe.dsp.compress import compress, decompress
from bwe.dsp.copyup import copy_up_hf
from bwe.dsp.stft import drop_nyquist, istft, pad_nyquist, stft
from bwe.losses import splice


def _pad_center(wave):
    """Zero-Pad um ``N_FFT`` vorne+hinten (rand-saubere STFT/iSTFT, = librosa center)."""
    return tf.pad(wave, [[cfg.N_FFT, cfg.N_FFT]])


def _crop_center(wave, orig_len):
    """Entfernt das Center-Padding und schneidet exakt auf ``orig_len`` zurück."""
    return wave[cfg.N_FFT : cfg.N_FFT + orig_len]


def reconstruct_copyup(
    input_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    energy_match: bool = True,
    center: bool = True,
) -> tf.Tensor:
    """Bandbegrenzte Eingangs-Wellenform → Vollband-Rekonstruktion (Wellenform).

    Erwartet ein bereits bandbegrenztes Signal (das echte Tiefband). Das HF wird
    per Copy-Up im komprimierten Bereich erzeugt; das LF bleibt unangetastet.
    ``center=True`` macht die Ränder sauber und die Länge gleich der Eingangslänge.
    """
    input_wave = tf.convert_to_tensor(input_wave, tf.float32)
    orig_len = tf.shape(input_wave)[0]
    if center:
        input_wave = _pad_center(input_wave)
    spec = stft(input_wave)                                   # [513, T] complex

    comp = drop_nyquist(compress(spec))                       # [512, T] komprimiert
    cu = copy_up_hf(comp, cutoff_bin=cutoff_bin, energy_match=energy_match)
    hf = decompress(cu)                                       # [512, T]

    lf = drop_nyquist(spec)[:cutoff_bin]                      # [cutoff, T] LF bit-genau
    spliced = tf.concat([lf, hf[cutoff_bin:]], axis=0)        # [512, T]
    out = istft(pad_nyquist(spliced))                         # Vollband-Wellenform
    return _crop_center(out, orig_len) if center else out


def baseline_from_fullband(
    fullband_wave,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    energy_match: bool = True,
    center: bool = True,
):
    """Komfort für Eval/Demo: Vollband-Signal bandbegrenzen, dann rekonstruieren.

    Returns ``(pred_wave, input_wave)`` — die Rekonstruktion und der bandbegrenzte
    Eingang (z. B. für das Spektrogramm-Tripel). ``center=True`` umschließt die
    **ganze** Kette (bandlimit **und** Rekonstruktion), damit beide Ausgaben
    randsauber und exakt so lang wie das Vollband-Eingangssignal sind.
    """
    fullband_wave = tf.convert_to_tensor(fullband_wave, tf.float32)
    orig_len = tf.shape(fullband_wave)[0]
    if center:
        fullband_wave = _pad_center(fullband_wave)
    input_wave = bandlimit(fullband_wave, cutoff_bin)
    pred = reconstruct_copyup(input_wave, cutoff_bin, energy_match, center=False)
    if center:
        input_wave = _crop_center(input_wave, orig_len)
        pred = _crop_center(pred, orig_len)
    return pred, input_wave


def _model_input(input_wave, cutoff_bin):
    """Bandbegrenzte Wellenform → Modelleingang ``[512, T, 3]`` (komprimiert, Copy-Up-gefüllt)."""
    spec = stft(input_wave)
    cu = copy_up_hf(drop_nyquist(compress(spec)), cutoff_bin=cutoff_bin)       # [512, T] complex
    ri = tf.stack([tf.math.real(cu), tf.math.imag(cu)], axis=-1)              # [512, T, 2]
    freq = tf.cast(tf.linspace(0.0, 1.0, cfg.N_BINS_NET), tf.float32)[:, None, None]
    freq = tf.tile(freq, [1, tf.shape(ri)[1], 1])                            # [512, T, 1]
    return spec, tf.concat([ri, freq], axis=-1)                              # [512, T, 3]


def reconstruct_model(generator, input_wave, cutoff_bin: int = cfg.CUTOFF_BIN,
                      center: bool = True) -> tf.Tensor:
    """Vollband-Rekonstruktion mit trainiertem Generator: HF gelernt, LF bit-genau gesplict."""
    input_wave = tf.convert_to_tensor(input_wave, tf.float32)
    orig_len = tf.shape(input_wave)[0]
    if center:
        input_wave = _pad_center(input_wave)
    spec, inp = _model_input(input_wave, cutoff_bin)
    out = generator(inp[tf.newaxis], training=False)                          # [1, 512, T, 2]
    spliced = splice(out, inp[tf.newaxis], cutoff_bin)[0]                     # LF=inp, HF=Modell
    pred = decompress(tf.complex(spliced[..., 0], spliced[..., 1]))           # [512, T] complex
    lf = drop_nyquist(spec)[:cutoff_bin]                                      # original LF bit-genau
    full = tf.concat([lf, pred[cutoff_bin:]], axis=0)
    wave = istft(pad_nyquist(full))
    return _crop_center(wave, orig_len) if center else wave


def model_from_fullband(generator, fullband_wave, cutoff_bin: int = cfg.CUTOFF_BIN,
                        center: bool = True):
    """Wie :func:`baseline_from_fullband`, aber mit dem trainierten Modell."""
    fullband_wave = tf.convert_to_tensor(fullband_wave, tf.float32)
    orig_len = tf.shape(fullband_wave)[0]
    if center:
        fullband_wave = _pad_center(fullband_wave)
    input_wave = bandlimit(fullband_wave, cutoff_bin)
    pred = reconstruct_model(generator, input_wave, cutoff_bin, center=False)
    if center:
        input_wave = _crop_center(input_wave, orig_len)
        pred = _crop_center(pred, orig_len)
    return pred, input_wave
