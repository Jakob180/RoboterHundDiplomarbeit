"""Gemeinsame Einstellungen für alle Tests (pytest lädt diese Datei automatisch)."""

import os
import sys
from pathlib import Path

# Tests dürfen nie echte Servos bewegen
os.environ["NOVA_SIM"] = "1"
# Tests rechnen mit den Platzhalter-Werten aus servos.py, nicht mit der Kalibrierung des echten Hundes
os.environ["NOVA_CALIBRATION"] = str(Path(__file__).resolve().parent / "keine-kalibrierung.json")

ROOT = Path(__file__).resolve().parent.parent
for folder in ("src", "model"):
    sys.path.insert(0, str(ROOT / folder))
