"""Export eines trainierten Modells nach ONNX (braucht torch, stable-baselines3, onnxruntime)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

from stable_baselines3 import PPO  # noqa: E402

import export_onnx  # noqa: E402
import policy  # noqa: E402
from nova_env import NovaEnv  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Ein kleines, untrainiertes Modell, so gespeichert wie von train.py."""
    run = tmp_path_factory.mktemp("lauf")
    env = NovaEnv(randomize=False, pushes=False)
    model = PPO("MlpPolicy", env, device="cpu", seed=0,
                policy_kwargs=dict(net_arch=dict(pi=[16, 16], vf=[16]), log_std_init=-1.0))
    model.save(run / "final.zip")
    (run / "settings.json").write_text(json.dumps(policy.settings(env.policy)), encoding="utf-8")
    return run / "final.zip", model


def test_export_writes_network_and_settings(trained, tmp_path):
    zip_path, _ = trained
    onnx_path = export_onnx.export(zip_path, tmp_path / "nova_test.onnx")
    assert onnx_path.exists()
    meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["mode"] == "residual"
    assert meta["source"] == f"{zip_path.parent.name}/final.zip"  # Lauf + Datei, ohne Pfad davor


def test_onnx_gives_the_same_actions_as_the_trained_model(trained, tmp_path):
    zip_path, model = trained
    onnx_path = export_onnx.export(zip_path, tmp_path / "nova_test.onnx")
    assert export_onnx.check(zip_path, onnx_path) < 1e-5
    net = policy.OnnxNetwork(onnx_path)
    obs = np.random.default_rng(3).normal(size=34).astype(np.float32)
    expected, _ = model.predict(obs, deterministic=True)
    assert net(obs) == pytest.approx(np.clip(expected, -1, 1), abs=1e-5)


def test_export_without_settings_from_training_is_refused(tmp_path):
    env = NovaEnv(randomize=False, pushes=False)
    PPO("MlpPolicy", env, device="cpu", policy_kwargs=dict(net_arch=[8])).save(tmp_path / "x.zip")
    with pytest.raises(FileNotFoundError, match="settings.json"):
        export_onnx.export(tmp_path / "x.zip", tmp_path / "x.onnx")
