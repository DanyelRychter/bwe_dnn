# bwe_dnn — Bandwidth Extension für Musik

Deep-Learning-Abschlussprojekt (alfatraining). Ein bandbegrenztes Musiksignal
bekommt fehlende **hohe Frequenzen** zurück. Aufbau in Stufen:
**Copy-Up** (DSP-Baseline) → **komplexe Regression** (U-Net) → **GAN** (PatchGAN).

> Das Problem ist *ill-posed*: Aus dem Tiefband ist das fehlende HF nicht eindeutig
> rekonstruierbar. Regression mittelt → verwaschen; das GAN modelliert die Verteilung
> plausibler HF → scharf. Dieser Kontrast ist die zentrale Story.

## Status

**Phase A (DSP-Infrastruktur) — fertig & getestet.** STFT-Kette, Power-Law-Kompression,
Bandbegrenzung, Copy-Up und die `tf.data`-Pipeline stehen, alle mit Mini-Tests.

**Phase B (Copy-Up-Baseline + Evaluation) — fertig & getestet.** End-to-End-
Rekonstruktion, Metriken (LSD-HF, SI-SDR) und Präsentations-Plots. Baseline-Referenz:
Copy-Up senkt die LSD-HF drastisch, verschlechtert aber die SI-SDR — die
„Metrik vs. Ohr"-Divergenz zeigt sich schon hier.

**Phase C (komplexe Regression, Stufe 1) — Code fertig & lokal abgesichert.** 2D-U-Net-
Generator, RI+Mag-Loss (HF-maskiert) + Splicing, Keras-Trainingsloop mit Checkpointing/
Resume. Pipeline um deterministischen Val/Test-Modus + On-the-fly-Resampling (für Kaggle)
erweitert. Lokaler Overfit-ein-Batch-Test besteht. **Subset-/Volltraining läuft auf Kaggle**
(`notebooks/kaggle_train_regression.ipynb`).

## Projektstruktur

```
bwe/
  config.py            # alle Hyperparameter + Pfade + abgeleitete Werte (Bins etc.)
  dsp/
    stft.py            # STFT/iSTFT, drop/pad Nyquist (513 <-> 512)
    compress.py        # Power-Law-Kompression (nur Magnitude)
    bandlimit.py       # STFT-Brickwall-Tiefpass @ 8 kHz
    copyup.py          # naive spektrale Hochkopie (HF-Initialisierung)
  data/
    splits.py          # kanonischer 86/14/50-Split (eigenständig, ohne ffmpeg)
    augment.py         # Stem-Remix, Gain, Polarität (auf dem Target)
    loaders.py         # load_demo: Track-Ausschnitt (mix/Stems) für Notebooks
    pipeline.py        # tf.data: Train (Augment) + Val/Test (deterministisch) + Resample
  models/
    generator.py       # 2D-U-Net-Generator (voll-faltend, Zeit-Pad/Crop)
  losses.py            # splice (LF original / HF Modell) + RI+Mag-Loss (HF)
  train/
    regression.py      # Stufe-1-Training (Keras-Subklasse; overfit/subset/full)
    checkpointing.py   # BackupAndRestore/ModelCheckpoint/CSVLogger/EarlyStopping
  infer/
    reconstruct.py     # Copy-Up- + Modell-Rekonstruktion (Audio -> Rekonstruktion)
  eval/
    metrics.py         # LSD-HF (Welle + spektral), SI-SDR
    plots.py           # Spektrogramm-Tripel, Crossover-Zoom
  tests/               # Mini-Tests pro Baustein
scripts/
  prepare_sample.py    # Sample-Tracks aus der ZIP -> 32-kHz-Cache (ohne Vollentpacken)
  check_dataset.py     # Struktur-/Integritätscheck des Caches
notebooks/
  01_dsp.ipynb                 # DSP-Demo (Round-Trip, Bandlimit, Copy-Up)
  02_copyup_baseline.ipynb     # Copy-Up-Baseline: Tripel, Audio, LSD-HF/SI-SDR
  03_regression.ipynb          # Stufe 1: Architektur + Ergebnisse (nach Kaggle)
  kaggle_train_regression.ipynb # Regressions-Training auf Kaggle (GPU)
```

## Setup

Genutzt wird das bestehende venv des Kurses (`../../venv`, TensorFlow 2.21).

```powershell
# Abhängigkeiten + Paket editierbar installieren
..\..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\..\venv\Scripts\python.exe -m pip install -e .

# Sample-Daten bereitstellen (liest direkt aus musdb18hq.zip, resampelt auf 32 kHz)
..\..\venv\Scripts\python.exe scripts\prepare_sample.py
..\..\venv\Scripts\python.exe scripts\check_dataset.py

# Tests
..\..\venv\Scripts\python.exe -m pytest bwe\tests -q
```

## Eckdaten (siehe `bwe/config.py`)

| | |
|---|---|
| Samplerate | 32 kHz (Nyquist 16 kHz) |
| STFT | `n_fft=1024`, `hop=256`, Hann; 513 → 512 Bins |
| Cutoff | 8 kHz (Bin 256) |
| Kompression | Power-Law `c = 0.3` (nur Magnitude) |
| Segment | 128 Frames ≈ 1,05 s |

## Daten & Hardware

* **MUSDB18-HQ** (unkomprimierte WAV-Stems, 44,1 kHz). Wird **nicht** vollständig
  entpackt — `prepare_sample.py` liest gezielt einzelne Tracks aus der ZIP und cacht
  sie als 32-kHz-WAVs außerhalb von OneDrive (`BWE_DATA_ROOT`).
* Lokal (Tablet) nur Debugging/Sanity-Checks; echtes Training läuft auf **Kaggle**
  (MUSDB18-HQ dort als Dataset einbindbar). Der `bwe`-Code ist dafür als Paket
  installierbar.
* Split kanonisch über die 14 vorgegebenen Validation-Tracks (in `splits.py`
  repliziert) — ein evtl. vorhandener Kaggle-`val/`-Ordner wird ignoriert (Leakage).
