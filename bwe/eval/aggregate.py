"""Aggregat-Auswertung über einen ganzen Split (Schritt 17b).

Iteriert über alle Tracks eines Splits, rekonstruiert je Track jeden Ansatz
(Copy-Up-Baseline + beliebige trainierte Generatoren) und misst LSD-HF, SI-SDR und
optional SI-SDR-HF. Ergebnis ist ein **Long-Format-DataFrame** (eine Zeile pro
Track × Methode) — daraus lassen sich die Aggregat-Tabelle (:func:`summary`) *und*
Per-Track-Boxplots (:func:`bwe.eval.plots.per_track_boxplots`) speisen.

Test-Disziplin: Auf ``valid`` gefahren, solange das GAN nicht final ist; ``test``
erst nach dem finalen Abschluss, dann genau **einmal**. ``get_split`` liefert nur
Tracks mit allen Stems → ein nur teilweise gecachter Split wird automatisch auf die
vorhandenen Tracks eingeschränkt.
"""

from __future__ import annotations

import pandas as pd

from bwe.data.loaders import load_demo
from bwe.data.splits import get_split
from bwe.eval import metrics as M
from bwe.infer.reconstruct import baseline_from_fullband, model_from_fullband

# Anzeigenamen der Metrik-Spalten (für Tabelle/Plots)
METRIC_LABELS = {
    "lsd_hf": "LSD-HF [dB]",
    "si_sdr": "SI-SDR [dB]",
    "si_sdr_hf": "SI-SDR-HF [dB]",
}


def _metrics_row(pred, target, hf_sisdr: bool) -> dict:
    """LSD-HF/SI-SDR(/HF) einer Rekonstruktion gegen das Target."""
    row = {"lsd_hf": M.lsd_hf(pred, target), "si_sdr": M.si_sdr(pred, target)}
    if hf_sisdr:
        row["si_sdr_hf"] = M.si_sdr_hf(pred, target)
    return row


def evaluate_split(
    split: str,
    generators: dict | None = None,
    seconds: float | None = None,
    offset: float = 0.0,
    limit: int | None = None,
    hf_sisdr: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rekonstruiert je Track jeden Ansatz und misst die Metriken.

    Parameters
    ----------
    split : ``"valid"`` (Trockenlauf) oder ``"test"`` (Finale).
    generators : ``{name: keras.Model}``, z. B. ``{"Regression": g_reg, "GAN": g_gan}``.
        Copy-Up wird immer zusätzlich (ohne Modell) ausgewertet.
    seconds : Länge je Track (``None`` = ganzer Track). Kleiner Wert = schneller lokal.
    offset : Startzeit je Track in Sekunden.
    limit : nur die ersten ``limit`` Tracks (Smoke-Test).
    hf_sisdr : zusätzlich SI-SDR-HF (Hochpass) messen.

    Returns
    -------
    Long-Format ``DataFrame`` mit Spalten ``track, method, lsd_hf, si_sdr[, si_sdr_hf]``.
    """
    generators = generators or {}
    tracks = get_split(split)
    if limit is not None:
        tracks = tracks[:limit]

    rows = []
    for i, track in enumerate(tracks):
        _, target = load_demo(split, i, seconds=seconds, offset=offset)

        cu, _ = baseline_from_fullband(target)
        rows.append({"track": track.name, "method": "Copy-Up",
                     **_metrics_row(cu, target, hf_sisdr)})

        for name, gen in generators.items():
            pred, _ = model_from_fullband(gen, target)
            rows.append({"track": track.name, "method": name,
                         **_metrics_row(pred, target, hf_sisdr)})

        if verbose:
            print(f"  [{i + 1}/{len(tracks)}] {track.name}")

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregat-Tabelle ``mean ± std`` je Methode über alle Tracks.

    Zeilen = Methoden (Copy-Up zuerst, dann die Generatoren in Eingabereihenfolge),
    Spalten = Metriken als formatierte Strings ``"mean ± std"``. Die numerischen
    Per-Track-Werte bleiben im Long-DataFrame (für Boxplots).
    """
    metric_cols = [c for c in ("lsd_hf", "si_sdr", "si_sdr_hf") if c in df.columns]
    g = df.groupby("method", sort=False)
    out = {}
    for c in metric_cols:
        mean, std = g[c].mean(), g[c].std()
        out[METRIC_LABELS[c]] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean, std)]
    return pd.DataFrame(out, index=g[metric_cols[0]].mean().index)
