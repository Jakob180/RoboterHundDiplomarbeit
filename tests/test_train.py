"""Training (braucht stable-baselines3). Nur ein winziger Probelauf, kein echtes Lernen."""

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("tensorboard")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import config  # noqa: E402
import train  # noqa: E402


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("training")
    final = train.train(quick=True, run="probe", out=out, vec_cls=DummyVecEnv)
    return out, final


def test_quick_run_saves_the_final_model(quick_run):
    out, final = quick_run
    assert final == out / "checkpoints" / "probe" / "final.zip"
    model = PPO.load(final, device="cpu")
    action, _ = model.predict(np.zeros(34, dtype=np.float32), deterministic=True)
    assert action.shape == (12,)


def test_every_curriculum_stage_is_saved(quick_run):
    out, _ = quick_run
    for name, *_ in config.CURRICULUM["residual"]:
        assert (out / "checkpoints" / "probe" / f"{name}.zip").exists()


def test_settings_of_the_training_are_kept(quick_run):
    out, _ = quick_run
    meta = json.loads((out / "checkpoints" / "probe" / "settings.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "residual"
    assert meta["n_obs"] == 34


def test_learning_curves_contain_reward_terms_and_episodes(quick_run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out, _ = quick_run
    folders = [p for p in (out / "runs").iterdir() if p.is_dir()]
    assert len(folders) == 1  # alle Abschnitte in einer Kurve
    tags = EventAccumulator(str(folders[0])).Reload().Tags()["scalars"]
    assert "terme/tempo" in tags and "terme/neigung" in tags
    assert "nova/strecke" in tags and "nova/umgefallen" in tags


def test_stage_2_training_is_not_there_yet():
    with pytest.raises(ValueError, match="Phase 4"):
        train.train(mode="full", quick=True)


def test_next_steps_quote_paths_with_spaces():
    from pathlib import Path
    text = train.next_steps(Path("C:/Ki Projekte/RoboterHundDiplomarbeit/checkpoints/lauf/final.zip"))
    assert '--policy "C:/Ki Projekte/RoboterHundDiplomarbeit/checkpoints/lauf/final.zip"' in text
    assert 'export_onnx.py "C:/Ki Projekte/' in text


def test_last_curriculum_stage_pushes_up_to_25_newton():
    name, millions, randomize, push_force = config.CURRICULUM["residual"][-1]
    assert randomize and push_force == 25.0
    assert config.CURRICULUM["residual"][0][3] == 0.0  # erster Abschnitt ohne Stöße


def test_make_env_passes_the_foot_scale():
    env = train.make_env("residual", False, 0.0, 0, foot_scale=(20.0, 40.0, 15.0))
    assert env.policy.foot_scale == (20.0, 40.0, 15.0)


def test_foot_scale_is_an_option_of_the_command_line():
    assert train.parse_args([]).foot_scale == list(train.FOOT_SCALE)
    assert train.parse_args(["--foot-scale", "20", "40", "15"]).foot_scale == [20.0, 40.0, 15.0]


def test_settings_json_records_the_trained_foot_scale(tmp_path):
    train.train(quick=True, run="breit", out=tmp_path, vec_cls=DummyVecEnv, foot_scale=(20.0, 40.0, 15.0))
    meta = json.loads((tmp_path / "checkpoints" / "breit" / "settings.json").read_text(encoding="utf-8"))
    assert meta["scales"]["foot"] == [20.0, 40.0, 15.0]
