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


def test_stage_2_needs_a_cloned_network_to_start_from():
    with pytest.raises(ValueError, match="imitation.py"):
        train.train(mode="full", quick=True)


def test_next_steps_quote_paths_with_spaces():
    from pathlib import Path
    text = train.next_steps(Path("C:/Ki Projekte/RoboterHundDiplomarbeit/checkpoints/lauf/final.zip"))
    assert '--policy "C:/Ki Projekte/RoboterHundDiplomarbeit/checkpoints/lauf/final.zip"' in text
    assert 'export_onnx.py "C:/Ki Projekte/' in text


def test_last_curriculum_stage_pushes_up_to_25_newton():
    name, millions, randomize, push_force, _ = config.CURRICULUM["residual"][-1]
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


def test_push_training_uses_strong_side_pushes():
    from randomization import PUSH_STYLES
    (name, millions, randomize, push, train_actor), = config.PUSH_TRAINING
    assert randomize and push in PUSH_STYLES


def test_make_env_with_a_push_style():
    from randomization import PUSH_STYLES
    env = train.make_env("residual", True, "stark-seitlich", 0)
    assert env.pushes and env.push_style == PUSH_STYLES["stark-seitlich"]


def test_push_training_continues_a_trained_network(quick_run, tmp_path):
    _, source = quick_run
    final = train.train(quick=True, run="stoss", out=tmp_path, vec_cls=DummyVecEnv, start_from=source)
    folder = tmp_path / "checkpoints" / "stoss"
    assert final == folder / "final.zip"
    name = config.PUSH_TRAINING[0][0]
    assert (folder / f"{name}.zip").exists()
    assert not (folder / f"{config.CURRICULUM['residual'][0][0]}.zip").exists()  # kein Lehrplan von vorne
    assert json.loads((folder / "settings.json").read_text(encoding="utf-8")) == \
        json.loads((source.parent / "settings.json").read_text(encoding="utf-8"))


def test_push_training_keeps_the_foot_scale_of_the_network(quick_run, tmp_path):
    import shutil
    import policy
    _, source = quick_run
    wide = tmp_path / "breit"
    wide.mkdir()
    shutil.copy(source, wide / "final.zip")
    (wide / "settings.json").write_text(
        json.dumps(policy.settings(policy.ResidualMode(foot_scale=(20.0, 40.0, 15.0)))), encoding="utf-8",
    )
    train.train(quick=True, run="stoss-breit", out=tmp_path, vec_cls=DummyVecEnv, start_from=wide / "final.zip")
    meta = json.loads((tmp_path / "checkpoints" / "stoss-breit" / "settings.json").read_text(encoding="utf-8"))
    assert meta["scales"]["foot"] == [20.0, 40.0, 15.0]


def test_push_training_is_an_option_of_the_command_line():
    assert train.parse_args([]).push_training is None
    assert train.parse_args(["--push-training", "checkpoints/a/final.zip"]).push_training == "checkpoints/a/final.zip"


def test_stage_2_curriculum_warms_up_the_critic_first():
    first, *rest = config.CURRICULUM["full"]
    assert first.train_actor is False and first.millions < 1.0
    assert rest and all(stage.train_actor for stage in rest)
    assert sum(stage.millions for stage in config.CURRICULUM["full"]) >= 30.0  # Plan: 30 bis 100 Mio.


def test_imitation_fades_from_one_to_a_fifth():
    start, end = config.IMITATION_FADE
    assert (start, end) == (1.0, 0.2)
    assert train.imitation_weight(0, 100) == pytest.approx(1.0)
    assert train.imitation_weight(50, 100) == pytest.approx(0.6)
    assert train.imitation_weight(100, 100) == pytest.approx(0.2)
    assert train.imitation_weight(150, 100) == pytest.approx(0.2)


def test_frozen_actor_stays_while_the_critic_learns():
    import torch
    from nova_env import NovaEnv
    model = PPO("MlpPolicy", NovaEnv(mode="full", randomize=False, pushes=False), device="cpu", seed=0,
                n_steps=64, batch_size=64, n_epochs=2, policy_kwargs={"net_arch": [8]})
    actor = [t.detach().clone() for t in train.actor_parameters(model.policy)]
    critic = [t.detach().clone() for t in model.policy.mlp_extractor.value_net.parameters()]
    train.freeze_actor(model.policy, True)
    model.learn(128)
    assert all(torch.equal(a, b) for a, b in zip(actor, train.actor_parameters(model.policy)))
    assert not all(torch.equal(a, b) for a, b in zip(critic, model.policy.mlp_extractor.value_net.parameters()))
    train.freeze_actor(model.policy, False)
    model.learn(128)
    assert not all(torch.equal(a, b) for a, b in zip(actor, train.actor_parameters(model.policy)))


@pytest.fixture(scope="module")
def cloned_start(tmp_path_factory):
    import imitation
    out = tmp_path_factory.mktemp("bc")
    obs, actions = imitation.collect(500, seed=0, workers=1)
    final, _ = imitation.behavior_cloning(obs, actions, out / "bc", epochs=2)
    return final


def test_stage_2_training_continues_the_cloned_network(cloned_start, tmp_path):
    final = train.train(quick=True, run="voll", out=tmp_path, vec_cls=DummyVecEnv, start_from=cloned_start)
    folder = tmp_path / "checkpoints" / "voll"
    assert final == folder / "final.zip"
    for stage in config.CURRICULUM["full"]:
        assert (folder / f"{stage.name}.zip").exists()
    meta = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "full"


def test_start_from_is_an_option_of_the_command_line():
    assert train.parse_args([]).start_from is None
    assert train.parse_args(["--start-from", "checkpoints/full-bc/final.zip"]).start_from == "checkpoints/full-bc/final.zip"


def test_stage_2_starts_with_little_random_trying(cloned_start, tmp_path):
    import torch
    train.train(quick=True, run="leise", out=tmp_path, vec_cls=DummyVecEnv, start_from=cloned_start)
    warm = PPO.load(tmp_path / "checkpoints" / "leise" / f"{config.CURRICULUM['full'][0].name}.zip", device="cpu")
    assert torch.allclose(warm.policy.log_std, torch.full((12,), config.FULL_LOG_STD_INIT))
    assert config.FULL_LOG_STD_INIT < config.LOG_STD_INIT
