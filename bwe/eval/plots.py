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


def spectro_grid6(
    target, band, cu, reg, gan,
    vmin: float = -80, vmax: float = 20, diff_lim: float = 30.0,
    suptitle: str | None = None,
):
    """2×3-Raster für den Modellvergleich (Präsentation).

    Oben: Original / Bandbegrenzt / Copy-Up — unten: Regression / GAN /
    Differenz GAN − Original (dB). Jedes Argument = Wellenform oder Spektrogramm.
    Die Differenz nutzt eine divergierende Skala (rot = zu laut, blau = zu leise).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    panels = [(target, "Original"), (band, "Bandbegrenzt (Input)"), (cu, "Copy-Up"),
              (reg, "Regression"), (gan, "GAN")]
    for ax, (x, ttl) in zip(axes.ravel(), panels):
        im = show_spec(ax, x, ttl, vmin, vmax)
    for ax in axes[0]:                      # x-Label nur unten (Titel-Kollision vermeiden)
        ax.set_xlabel("")
    fig.colorbar(im, ax=axes[:, :2], label="dB")

    ax = axes[1, 2]
    d = to_db(gan) - to_db(target)
    im2 = ax.imshow(d, origin="lower", aspect="auto", cmap="coolwarm",
                    vmin=-diff_lim, vmax=diff_lim,
                    extent=[0, d.shape[1], 0, cfg.SR / 2 / 1000])
    ax.axhline(cfg.CUTOFF_HZ / 1000, color="k", lw=0.8, ls="--")
    ax.set_title("Differenz GAN − Original"); ax.set_xlabel("Frame"); ax.set_ylabel("kHz")
    fig.colorbar(im2, ax=axes[:, 2], label="Δ dB")
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def frame_response(items, t0: int, title: str | None = None,
                   ylim_bottom: float = -50, mark_bands: bool = False):
    """Frequenzgang **eines** STFT-Frames: mehrere Versionen übereinander.

    ``items`` = Liste ``(label, spec_oder_wave)`` oder ``(label, x, plot_kwargs)``;
    Spektrogramme werden bei ``t0`` geschnitten, Wellenformen vorher transformiert.
    ``mark_bands=True`` hebt das Copy-Up-Quellband (4–8 kHz) + Patch-Grenzen hervor.
    """
    freqs = np.arange(cfg.N_BINS_NET) * cfg.FREQ_RES / 1000    # Bin -> kHz
    fig, ax = plt.subplots(figsize=(13, 4))
    if mark_bands:
        ax.axvspan(cfg.COPYUP_SRC_LO_BIN * cfg.FREQ_RES / 1000,
                   cfg.CUTOFF_HZ / 1000, color="C0", alpha=0.08)
        for f in (4, 8, 12):
            ax.axvline(f, color="grey", ls="--", lw=0.8)
    else:
        ax.axvline(cfg.CUTOFF_HZ / 1000, color="grey", ls="--", lw=0.8)
    for item in items:
        label, x = item[0], item[1]
        kw = dict(item[2]) if len(item) > 2 else {}
        S = to_db(x)[:cfg.N_BINS_NET]
        ax.plot(freqs, S[:, t0], label=label, **kw)
    ax.set_xlim(0, cfg.SR / 2 / 1000)
    ax.set_ylim(bottom=ylim_bottom)
    ax.set_xlabel("Frequenz [kHz]"); ax.set_ylabel("Magnitude [dB]")
    ax.set_title(title or f"Frequenzgang eines einzelnen STFT-Frames (t = {t0})")
    ax.legend(loc="lower left", fontsize=8)
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


def split_boxplots(dfs: dict, metrics=("lsd_hf", "si_sdr"), method_order=None):
    """Kombinierte Boxplots über mehrere Splits (Overfitting-Analyse, ein Bild).

    ``dfs`` = ``{"train": df, "valid": df, "test": df}`` (Long-Format aus
    :func:`bwe.eval.aggregate.evaluate_split`). Ein Panel je Metrik; auf der x-Achse
    die Methoden, innerhalb jeder Methoden-Gruppe eine Box je Split (farbcodiert,
    direkt nebeneinander). Train ≈ Test ⇒ das Modell generalisiert.
    """
    from bwe.eval.aggregate import METRIC_LABELS

    splits = list(dfs)
    first = dfs[splits[0]]
    metrics = [m for m in metrics if m in first.columns]
    methods = method_order or list(dict.fromkeys(first["method"]))
    n_s = len(splits)
    width = 0.8 / n_s                                   # Boxbreite innerhalb der Gruppe
    colors = [f"C{i}" for i in range(n_s)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for j, (split, color) in enumerate(zip(splits, colors)):
            data = [dfs[split].loc[dfs[split]["method"] == m, metric].to_numpy()
                    for m in methods]
            pos = [i + (j - (n_s - 1) / 2) * width for i in range(len(methods))]
            bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                            showmeans=True, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(color); box.set_alpha(0.45)
            for med in bp["medians"]:
                med.set_color(color)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("dB")
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.45) for c in colors]
    axes[-1].legend(handles, splits, loc="best", fontsize=9)
    fig.tight_layout()
    return fig
