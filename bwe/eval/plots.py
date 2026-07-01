"""Plots für die Präsentation: Spektrogramm-Tripel und Crossover-Zoom.

Die Funktionen akzeptieren je Panel **entweder eine Wellenform** (1D → STFT wird
intern berechnet) **oder ein fertiges Spektrogramm** ``[F, T]`` (z. B. das
komprimierte Input-Spektrogramm aus dem DSP-Notebook). So sind dieselben Helfer in
beiden Notebooks nutzbar.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bwe import config as cfg
from bwe.dsp.stft import stft


def to_db(x, eps: float = 1e-6) -> np.ndarray:
    """Log-Magnitude in dB. ``x`` = Wellenform (1D) **oder** Spektrogramm ``[F, T]``."""
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    if arr.ndim == 1:                       # Wellenform -> STFT
        arr = stft(arr).numpy()
    return 20.0 * np.log10(np.abs(arr) + eps)


# Rückwärtskompatibler Alias (Wellenform -> dB-Spektrogramm)
def logmag(wave, eps: float = 1e-6) -> np.ndarray:
    return to_db(wave, eps)


def show_spec(ax, x, title: str = "", vmin: float = -80, vmax: float = 20,
              draw_cutoff: bool = True, bin_range=None):
    """Ein Spektrogramm-Panel zeichnen. ``x`` = Wellenform oder Spektrogramm.

    ``bin_range=(lo, hi)`` zeigt nur diesen Frequenz-Bin-Bereich (für den Zoom).
    """
    S = to_db(x)
    if bin_range is not None:
        lo, hi = bin_range
        S = S[lo:hi]
        y0, y1 = lo * cfg.FREQ_RES / 1000, hi * cfg.FREQ_RES / 1000
    else:
        y0, y1 = 0.0, cfg.SR / 2 / 1000
    im = ax.imshow(S, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                   extent=[0, S.shape[1], y0, y1])
    if draw_cutoff:
        ax.axhline(cfg.CUTOFF_HZ / 1000, color="w", lw=0.8, ls="--")
    ax.set_title(title); ax.set_xlabel("Frame"); ax.set_ylabel("kHz")
    return im


def spectro_triple(
    panel_a, panel_b, panel_c,
    titles=("Bandbegrenzt (Input)", "Rekonstruktion", "Original (Target)"),
    vmin: float = -80, vmax: float = 20, draw_cutoff: bool = True,
):
    """Drei Spektrogramme nebeneinander (je Wellenform oder Spektrogramm)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, x, t in zip(ax, (panel_a, panel_b, panel_c), titles):
        im = show_spec(a, x, t, vmin, vmax, draw_cutoff)
    fig.colorbar(im, ax=ax, label="dB")
    return fig


def crossover_zoom(
    panel_a, panel_b,
    cutoff_bin: int = cfg.CUTOFF_BIN, half_bins: int = 64,
    titles=("Rekonstruktion", "Original"),
    vmin: float = -80, vmax: float = 20,
):
    """Zoom um den Cutoff (die „Naht"). Ohne Cutoff-Linie — sie würde die Naht verdecken."""
    rng = (cutoff_bin - half_bins, cutoff_bin + half_bins)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, x, ttl in zip(ax, (panel_a, panel_b), titles):
        show_spec(a, x, ttl, vmin, vmax, draw_cutoff=False, bin_range=rng)
    fig.tight_layout()
    return fig


def per_track_boxplots(df, metrics=("lsd_hf", "si_sdr"), method_order=None):
    """Boxplots je Metrik über **alle Tracks** eines Splits, gruppiert nach Methode.

    ``df`` = Long-Format aus :func:`bwe.eval.aggregate.evaluate_split` (Spalten
    ``method`` + Metriken). Ein Panel je Metrik; pro Panel eine Box je Methode
    (Copy-Up/Regression/GAN) — zeigt Streuung und Ausreißer, nicht nur den
    Mittelwert der Aggregat-Tabelle.
    """
    from bwe.eval.aggregate import METRIC_LABELS

    metrics = [m for m in metrics if m in df.columns]
    methods = method_order or list(dict.fromkeys(df["method"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        data = [df.loc[df["method"] == m, metric].to_numpy() for m in methods]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("dB")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
