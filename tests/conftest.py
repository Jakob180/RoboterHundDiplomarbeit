"""Gemeinsame Einstellungen für alle Tests (pytest lädt diese Datei automatisch)."""

import os
import sys
from pathlib import Path

# Tests dürfen nie echte Servos bewegen
os.environ["NOVA_SIM"] = "1"

ROOT = Path(__file__).resolve().parent.parent
for folder in ("src", "model"):
    sys.path.insert(0, str(ROOT / folder))
