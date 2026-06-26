"""Pytest-Setup: Umgebungsvariablen MÜSSEN vor dem TensorFlow-Import gesetzt sein.

``TF_ENABLE_ONEDNN_OPTS=0`` entschärft einen bekannten oneDNN-Crash auf dieser
Hardware; die Log-Level-Variable hält die Test-Ausgabe sauber.
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
