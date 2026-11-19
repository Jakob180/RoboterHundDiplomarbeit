"""
Trainiertes Modell (Stable-Baselines3, .zip) -> ONNX-Datei für den Pi (Plan Abschnitt 5.4).

Exportiert wird nur der "Actor", also der Teil des Netzes, der aus den 34
Eingängen die 12 Ausgaben rechnet (ohne Zufall). Daneben kommt eine .json
mit den Einstellungen aus dem Training (settings.json aus dem Trainingsordner).
Der Pi prüft beim Laden, dass sie zu seiner policy.py passen.

    python model/export_onnx.py checkpoints/<lauf>/final.zip --name nova_residual_v1

Danach vergleicht das Skript für 1.000 zufällige Eingänge, ob ONNX dasselbe
rechnet wie das Original (Abweichung unter 1e-5). Dafür braucht es onnxruntime;
das gibt es für Python 3.15 unter Windows noch nicht. Dann den Vergleich am
Laptop machen (gleicher Befehl, Ordner checkpoints/<lauf> mitnehmen).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policy import OBS_CLIP  # noqa: E402

POLICIES = Path(__file__).resolve().parent / "policies"


class _Actor(torch.nn.Module):
    """Nur der deterministische Teil: Eingänge -> Mittelwert der Aktion."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        p = self.policy
        return p.action_net(p.mlp_extractor.forward_actor(p.pi_features_extractor(obs)))


def _settings_of(zip_path: Path) -> dict:
    path = zip_path.parent / "settings.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt. train.py legt sie beim Start an, sie gehört zum Modell.")
    return json.loads(path.read_text(encoding="utf-8"))


def export(zip_path, onnx_path) -> Path:
    """Modell laden, Actor als ONNX speichern, Einstellungen als .json daneben."""
    zip_path, onnx_path = Path(zip_path), Path(onnx_path)
    meta = _settings_of(zip_path)
    model = PPO.load(zip_path, device="cpu")
    actor = _Actor(model.policy).eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)  # Hinweise zu torchvision u. ä. ausblenden
    with torch.no_grad():
        torch.onnx.export(
            actor, (torch.zeros(1, meta["n_obs"]),), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_shapes={"obs": {0: torch.export.Dim("batch")}},
            dynamo=True, verbose=False, external_data=False,
        )
    meta["source"] = zip_path.name
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return onnx_path


def check(zip_path, onnx_path, n: int = 1000) -> float:
    """Größte Abweichung zwischen Original und ONNX bei n zufälligen Eingängen."""
    import onnxruntime

    model = PPO.load(Path(zip_path), device="cpu")
    actor = _Actor(model.policy).eval()
    rng = np.random.default_rng(0)
    obs = np.clip(rng.normal(size=(n, model.observation_space.shape[0])), -OBS_CLIP, OBS_CLIP).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs)).numpy()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["action"], {"obs": obs})[0]
    return float(np.max(np.abs(got - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiertes Modell als ONNX für den Pi speichern")
    parser.add_argument("model", help="z. B. checkpoints/<lauf>/final.zip")
    parser.add_argument("--name", help="Dateiname ohne Endung, Standard: nova_<modus>_v1")
    args = parser.parse_args()

    zip_path = Path(args.model)
    name = args.name or f"nova_{_settings_of(zip_path)['mode']}_v1"
    onnx_path = export(zip_path, POLICIES / f"{name}.onnx")
    print(f"Gespeichert: {onnx_path} (+ .json)")
    try:
        difference = check(zip_path, onnx_path)
    except ImportError:
        print("Vergleich übersprungen: onnxruntime fehlt (Python 3.15 unter Windows). Am Laptop wiederholen.")
        return
    status = "OK" if difference < 1e-5 else "FEHLER, Netz nicht verwenden"
    print(f"Vergleich Original vs. ONNX (1.000 Eingänge): größte Abweichung {difference:.1e} -> {status}")


if __name__ == "__main__":
    main()
