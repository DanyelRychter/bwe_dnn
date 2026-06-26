"""Copy-Up — naive spektrale Hochkopie als HF-Initialisierung (Stufe 0).

Bewusst **nicht** SBR/IGF: kein Codec-Verfahren mit Metadaten, nur lose vom
Grundgedanken inspiriert und deutlich simpler. Das Band **4–8 kHz** (Bins 128..256)
wird zweimal nach oben gekachelt (→ 8–12 kHz und 12–16 kHz). Spektrale
*Translation* (nicht Spiegelung) erhält die harmonische Aufwärtsstruktur; die
kopierten Bins bringen ihre originale Phase mit (mehr Inter-Frame-Kohärenz als
Griffin-Lim).

Das untere Tiefband (0–4 kHz) wird bewusst **nicht** kopiert — Bass-Grundtöne nach
oben kopiert klingen unnatürlich. ``energy_match`` gleicht den Pegel grob am
Crossover an; ohne diesen Schritt ist der hörbare Pegelsprung („Crossover-Naht")
eine gute Demo.

Bewusst primitiv: Copy-Up soll die *unzureichende* Baseline sein, deren Kontrast
die gelernten Stufen sichtbar machen.
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg


def copy_up_hf(
    spec,
    energy_match: bool = True,
    ref_width: int = 16,
    cutoff_bin=None,
    src_lo_bin=None,
) -> tf.Tensor:
    """Füllt die HF-Bins (``>= cutoff_bin``) eines Spektrogramms ``[F, T]`` per Hochkopie.

    Parameters
    ----------
    spec : komplexes Spektrogramm ``[F, T]`` (LF gefüllt, HF leer).
    energy_match : Pegel der Kopie am Crossover an die LF-Hüllkurve angleichen.
    ref_width : Anzahl Bins für die Pegel-Referenz beidseits des Crossovers.
    cutoff_bin : Grenz-Bin (Default ``cfg.CUTOFF_BIN``). Darf ein Skalar-Tensor sein
        (variabler Cutoff pro Beispiel).
    src_lo_bin : Unterkante des Quellbands (Default = Oktave unter dem Cutoff,
        ``cutoff_bin // 2``). Das Quellband ``[src_lo_bin, cutoff_bin]`` wird so oft
        wie nötig nach oben gekachelt, um das HF zu füllen.
    """
    spec = tf.convert_to_tensor(spec, dtype=tf.complex64)
    if cutoff_bin is None:
        cutoff_bin = cfg.CUTOFF_BIN
    if src_lo_bin is None:
        src_lo_bin = cutoff_bin // 2

    lo, hi = src_lo_bin, cutoff_bin
    width = hi - lo
    cutoff = cutoff_bin

    n_bins = tf.shape(spec)[0]
    n_hf = n_bins - cutoff
    n_tiles = (n_hf + width - 1) // width

    src = spec[lo:hi]                                        # [width, T] Quellband 4–8 kHz
    tiled = tf.tile(src, [n_tiles, 1])[:n_hf]               # [n_hf, T] hochkopiert
    lf = spec[:cutoff]                                       # [cutoff, T] erhaltenes LF

    if energy_match:
        # Pegel knapp unter dem Cutoff (LF-Rand) als Ziel, Pegel am Kopie-Anfang als Ist.
        ref = tf.reduce_mean(tf.abs(spec[cutoff - ref_width:cutoff]), axis=0, keepdims=True)
        cur = tf.reduce_mean(tf.abs(tiled[:ref_width]), axis=0, keepdims=True)
        gain = ref / (cur + cfg.COMPRESS_EPS)               # [1, T] per Frame
        tiled = tiled * tf.cast(gain, spec.dtype)

    return tf.concat([lf, tiled], axis=0)
