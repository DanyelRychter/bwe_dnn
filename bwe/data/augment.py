"""Augmentation — IMMER auf dem TARGET, der Input wird danach abgeleitet.

Unabhängiges Augmentieren von Input und Target zerstört die Paarung. Deshalb
mischen wir hier nur das Vollband-Target und erzeugen den bandbegrenzten Input
erst anschließend daraus (siehe ``pipeline.py``).

Enthalten (alle billig pro Schritt):
* **Stem-Remix** — vocals/drums/bass/other mit Zufallsgains neu mischen
  (stärkster Hebel: kombinatorisch viele „neue" Mixe).
* **Gain** — Gesamtpegel skalieren (perzeptiv ~neutral).
* **Polarität** — Vorzeichen kippen (perzeptiv neutral, verändert die Wellenform).

Bewusst NICHT: Pitch-Shift (drückt Inhalt Richtung Nyquist → verschmutzt HF) und
Time-Stretch (Phase-Vocoder-Artefakte).
"""

from __future__ import annotations

import numpy as np

# Standard-Bereiche (linear)
STEM_GAIN_RANGE = (0.25, 1.25)   # pro Stem
MASTER_GAIN_RANGE = (0.5, 1.0)   # Gesamtpegel
POLARITY_PROB = 0.5


def mix_stems(stems: dict, gains: dict | None = None) -> np.ndarray:
    """Summiert Stems (mono, gleiche Länge) zu einem Mix. ``gains=None`` → Gain 1.0 (= Original-Mix)."""
    keys = list(stems.keys())
    out = np.zeros_like(stems[keys[0]])
    for k in keys:
        g = 1.0 if gains is None else float(gains[k])
        out = out + g * stems[k]
    return out


def augment_target(
    stems: dict,
    rng: np.random.Generator | None = None,
    enabled: bool = True,
) -> np.ndarray:
    """Aus mono Stems ein augmentiertes Vollband-Target erzeugen.

    ``enabled=False`` liefert den unveränderten Mix (Summe der Stems) — nützlich
    für Validation/Test und Sanity-Checks.
    """
    if not enabled:
        return _safe_norm(mix_stems(stems))

    rng = rng or np.random.default_rng()
    gains = {k: rng.uniform(*STEM_GAIN_RANGE) for k in stems}
    target = mix_stems(stems, gains)
    target *= rng.uniform(*MASTER_GAIN_RANGE)
    if rng.random() < POLARITY_PROB:
        target = -target
    return _safe_norm(target)


def _safe_norm(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Skaliert herunter, falls die Spitze > ``peak`` ist (Clipping vermeiden)."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)
