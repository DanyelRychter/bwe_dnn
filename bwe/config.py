"""Zentrale Konfiguration — alle Hyperparameter und Pfade an EINER Stelle.

Importierbar als ``import bwe.config as cfg``. Abgeleitete Werte (Bin-Indizes,
Segmentlängen) werden hier einmal berechnet, damit DSP-Kette, Loss und Pipeline
garantiert dieselben Grenzen verwenden (eine halbe Bin-Verschiebung zwischen
Bandlimit/Splicing/Loss wäre ein schwer zu findender Bug).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Audio / Samplerate
# --------------------------------------------------------------------------- #
SR: int = 32_000          # Ziel-Samplerate (Nyquist = 16 kHz)
SR_RAW: int = 44_100      # Samplerate des MUSDB18-HQ-Quellmaterials
MONO: bool = True         # L+R zu Mono mitteln (Stereo bleibt v2)

# --------------------------------------------------------------------------- #
# STFT
# --------------------------------------------------------------------------- #
N_FFT: int = 1_024        # FFT-Länge
HOP: int = 256            # Hop (75 % Overlap)
WIN: str = "hann"         # Fensterfunktion

FREQ_RES: float = SR / N_FFT          # Hz pro Bin = 31.25
N_BINS_FULL: int = N_FFT // 2 + 1     # 513 (inkl. DC und Nyquist)
N_BINS_NET: int = N_BINS_FULL - 1     # 512 (Nyquist-Bin fürs Netz weggelassen)

# --------------------------------------------------------------------------- #
# Bandbegrenzung / Cutoff
# --------------------------------------------------------------------------- #
def cutoff_bin_for(hz: float) -> int:
    """Cutoff-Frequenz (Hz) -> Bin-Index der Grenze (LF: < bin, HF: >= bin)."""
    return round(hz / FREQ_RES)


# Aktive Cutoff-Frequenz(en) in Hz. Aktuell nur 8 kHz. Weitere Werte hier
# ergänzen (z. B. 4000) → die Pipeline zieht pro Beispiel zufällig einen Cutoff
# (= Augmentation „variabler Cutoff"). Alles Cutoff-abhängige (Bandlimit, Copy-Up)
# ist parametrisch. Hinweis: echtes Multi-Cutoff-*Training* braucht zusätzlich
# einen Cutoff-Masken-Kanal, damit das Modell weiß, ab wo zu generieren ist (v2).
CUTOFFS_HZ: tuple[int, ...] = (8_000,)
CUTOFF_HZ: int = CUTOFFS_HZ[0]                  # Standard-/Einzel-Cutoff
CUTOFF_BIN: int = cutoff_bin_for(CUTOFF_HZ)     # 256: Bins 0..255 = LF, 256..511 = HF


def copyup_source_band(cutoff_bin: int = CUTOFF_BIN) -> tuple[int, int]:
    """Copy-Up-Quellband = die Oktave unter dem Cutoff → (lo, hi) = (cutoff//2, cutoff)."""
    return cutoff_bin // 2, cutoff_bin


# Default-Quellband (8 kHz): 4–8 kHz = Bins 128..256
COPYUP_SRC_LO_BIN, COPYUP_SRC_HI_BIN = copyup_source_band()

# --------------------------------------------------------------------------- #
# Kompression
# --------------------------------------------------------------------------- #
COMPRESS_C: float = 0.3   # Power-Law-Exponent (nur Magnitude)
COMPRESS_EPS: float = 1e-8

# --------------------------------------------------------------------------- #
# Segmentierung (Training)
# --------------------------------------------------------------------------- #
SEG_FRAMES: int = 128                                 # T; durch 16 teilbar (4 U-Net-Ebenen)
SEG_SAMPLES: int = N_FFT + (SEG_FRAMES - 1) * HOP     # 33536 ≈ 1.05 s
UNET_DEPTH_FACTOR: int = 16                           # 2**4: F und T müssen Vielfache sein

# --------------------------------------------------------------------------- #
# Pfade
# --------------------------------------------------------------------------- #
# Rohdaten-ZIP (MUSDB18-HQ, 44.1 kHz) — nur fürs einmalige Caching.
RAW_ZIP: Path = Path(
    os.environ.get(
        "BWE_RAW_ZIP",
        r"C:\Users\danyr\OneDrive\Dokumente\alfatraining\DeepLearning\Projektarbeit\musdb18hq.zip",
    )
)

# 32-kHz-Cache, bewusst AUSSERHALB von OneDrive (Sync-/Plattendruck vermeiden).
DATA_ROOT: Path = Path(
    os.environ.get("BWE_DATA_ROOT", r"C:\Users\danyr\bwe_data\musdb18hq_32k")
)

# MUSDB18-Stems (mixture wird als Summe der Stems rekonstruiert)
STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "other")

# --------------------------------------------------------------------------- #
# Generator (2D-U-Net)
# --------------------------------------------------------------------------- #
N_INPUT_CHANNELS: int = 3                       # Re, Im, Freq-Koord (+1 Cutoff-Maske = v2)
N_OUTPUT_CHANNELS: int = 2                       # Re, Im (linear)
UNET_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)   # 4 Ebenen; Stride-2 je Ebene
LEAKY_SLOPE: float = 0.2

# --------------------------------------------------------------------------- #
# Loss (RI+Mag, nur HF)
# --------------------------------------------------------------------------- #
W_RE: float = 1.0
W_IM: float = 1.0
W_MAG: float = 1.0                               # Magnitudenterm (evtl. leicht höher testen)
LOSS_EPS: float = 1e-8

# --------------------------------------------------------------------------- #
# Diskriminator (PatchGAN) + GAN (Stufe 2)
# --------------------------------------------------------------------------- #
DISC_CHANNELS: tuple[int, ...] = (64, 128, 256)  # 3× Conv2D-s2 (kein BatchNorm im D)
DISC_USE_SPECTRAL_NORM: bool = True              # Spectral Norm stabilisiert das D-Training

# GAN-Loss-Gewichte (Leitfaden §10: L_recon dominant ≈1, λ_adv klein, λ_fm mittel).
# Startwerte zum Festklopfen auf dem Subset (Schritt 15).
LAMBDA_ADV: float = 0.05                          # 0.01–0.1
LAMBDA_FM: float = 10.0                           # 1–10 (höher = stabilisierend)

GAN_LR: float = 2e-4                              # Adam (G und D)
GAN_BETA_1: float = 0.5                           # GAN-typisch (Regression nutzt 0.9)
GAN_D_WARMUP_STEPS: int = 300                     # Diskriminator-Vorlauf, G eingefroren
GAN_N_CRITIC: int = 1                             # D-Updates je G-Update (1:1 reicht meist)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 16                            # Kaggle/GPU; lokal kleiner setzen
LR: float = 2e-4                                 # Adam
ADAM_BETA_1: float = 0.9                         # Regression (GAN nutzt 0.5)
EPOCHS: int = 100                                # mit EarlyStopping
EARLY_STOP_PATIENCE: int = 12
VAL_SEGMENTS_PER_TRACK: int = 8                  # feste Segmente je Val/Test-Track (dichte Abdeckung)
# steps_per_epoch wird zur Laufzeit aus Gesamtdauer/(B·Seg) berechnet (None = auto)
STEPS_PER_EPOCH = None

# Checkpoints/Logs bewusst AUSSERHALB von OneDrive (Schreibsperren/Resume).
CKPT_ROOT: Path = Path(
    os.environ.get("BWE_CKPT_ROOT", str(Path.home() / "bwe_runs"))
)

SEED: int = 1234


def hf_mask_1d():
    """Boolesche Maske der Länge N_BINS_NET: True für HF-Bins (>= CUTOFF_BIN)."""
    import numpy as np

    idx = np.arange(N_BINS_NET)
    return idx >= CUTOFF_BIN


def summary() -> str:
    """Kompakte Übersicht der wichtigsten abgeleiteten Werte (für Notebooks/Logs)."""
    return (
        f"SR={SR} Hz | N_FFT={N_FFT} HOP={HOP} | {FREQ_RES:.3f} Hz/Bin\n"
        f"Bins: full={N_BINS_FULL} net={N_BINS_NET} | "
        f"Cutoffs={CUTOFFS_HZ} Hz | Standard {CUTOFF_HZ} Hz -> Bin {CUTOFF_BIN}\n"
        f"Copy-Up-Quellband = Bins {COPYUP_SRC_LO_BIN}..{COPYUP_SRC_HI_BIN}\n"
        f"Segment: {SEG_FRAMES} Frames = {SEG_SAMPLES} Samples (~{SEG_SAMPLES / SR:.3f} s)\n"
        f"DATA_ROOT={DATA_ROOT}"
    )


if __name__ == "__main__":
    print(summary())
    assert CUTOFF_BIN == 256, CUTOFF_BIN
    assert N_BINS_NET % UNET_DEPTH_FACTOR == 0
    assert SEG_FRAMES % UNET_DEPTH_FACTOR == 0
    print("config OK")
