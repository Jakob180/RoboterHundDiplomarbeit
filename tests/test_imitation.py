"""Stufe 2, Schritte 1 und 2: Lehrer-Daten sammeln und nachmachen (Behavior Cloning)."""

import json

import numpy as np
import pytest

import imitation
import policy
from nova_env import NovaEnv

DT = 0.02


def test_full_action_brings_stage_2_to_the_given_angles():
    target = policy.STAND_ANGLES + np.tile([2.0, 15.0, -12.0], 4)
    m = policy.FullMode()
    for _ in range(10):  # höchstens 8° je Schritt
        m.act(policy.angles_to_full_action(target), DT)
    assert m.angles == pytest.approx(target, abs=1e-9)


def test_full_action_is_limited_to_minus_one_to_one():
    a = policy.angles_to_full_action(policy.STAND_ANGLES + 100.0)
    assert np.all(a == 1.0)


def test_environment_tells_the_last_imu_reading():
    env = NovaEnv(mode="full", randomize=False, pushes=False)
    env.reset(seed=0)
    up, gyro = env.last_imu
    assert len(up) == 3 and len(gyro) == 3
    assert up[2] > 0.95  # steht gerade


def test_teacher_without_network_is_the_classic_trot():
    teacher = imitation.Teacher()
    trot = policy.ResidualMode()
    trot.set_command(150.0, 0.0)
    for _ in range(100):
        a = teacher.action((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 150.0, 0.0)
        trot.act(np.zeros(12), DT)
        assert a == pytest.approx(policy.angles_to_full_action(trot.angles))


def test_teacher_uses_its_network():
    plain = imitation.Teacher().action((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 150.0, 0.0)
    pushed = imitation.Teacher(lambda obs: np.ones(12)).action((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 150.0, 0.0)
    assert not np.allclose(plain, pushed)


def test_teacher_keeps_step_with_the_environment():
    env = NovaEnv(mode="full", randomize=False, pushes=False)
    env.reset(seed=0)
    teacher = imitation.Teacher()
    for _ in range(300):  # mit Befehlswechseln
        up, gyro = env.last_imu
        a = teacher.action(up, gyro, env.policy.target_speed, env.policy.target_yaw_rate)
        env.step(a)
        assert teacher.mode.trot.phase == pytest.approx(env.policy.trot.phase)
        assert teacher.mode.speed == pytest.approx(env.policy.speed)


def test_student_executing_the_teacher_walks_like_the_trot():
    env = NovaEnv(mode="full", randomize=False, pushes=False)
    env.reset(seed=0, options={"command": (150.0, 0.0)})
    teacher = imitation.Teacher()
    while True:
        up, gyro = env.last_imu
        _, _, fallen, truncated, info = env.step(
            teacher.action(up, gyro, env.policy.target_speed, env.policy.target_yaw_rate))
        if fallen or truncated:
            break
    assert not fallen and info["nova"]["strecke"] > 0.8


def test_collect_gives_observation_and_teacher_action_pairs():
    obs, actions = imitation.collect(300, seed=0, workers=1)
    assert obs.shape == (300, 34) and actions.shape == (300, 12)
    assert obs.dtype == actions.dtype == np.float32
    assert np.all(np.isfinite(obs)) and np.all(np.abs(actions) <= 1.0)


def test_collect_is_repeatable():
    a = imitation.collect(200, seed=3, workers=1)
    b = imitation.collect(200, seed=3, workers=1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_collect_in_parallel_gives_the_requested_number(tmp_path):
    obs, actions = imitation.collect(250, seed=0, workers=2)
    assert obs.shape == (250, 34) and actions.shape == (250, 12)


@pytest.fixture(scope="module")
def cloned(tmp_path_factory):
    pytest.importorskip("stable_baselines3")
    out = tmp_path_factory.mktemp("bc")
    obs, actions = imitation.collect(2000, seed=0, workers=1)
    path, history = imitation.behavior_cloning(obs, actions, out, epochs=15)
    return path, history, obs, actions


def test_behavior_cloning_learns_the_teacher(cloned):
    path, history, _, _ = cloned
    assert history[-1] < 0.5 * history[0]  # Fehler auf den Prüfdaten sinkt deutlich


def test_cloned_network_is_saved_like_a_trained_one(cloned):
    from stable_baselines3 import PPO
    path, _, obs, actions = cloned
    assert path.name == "final.zip"
    meta = json.loads((path.parent / "settings.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "full"
    policy.check_settings(meta)
    model = PPO.load(path, device="cpu")
    predicted, _ = model.predict(obs[:100], deterministic=True)
    assert predicted.shape == (100, 12)


def test_command_line_options():
    args = imitation.parse_args(["--teacher", "checkpoints/a/final.zip", "--samples", "5000"])
    assert args.teacher == "checkpoints/a/final.zip" and args.samples == 5000
    assert imitation.parse_args([]).teacher is None
