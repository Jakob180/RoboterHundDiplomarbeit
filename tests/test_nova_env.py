"""Trainingsumgebung (Gymnasium)."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import config
from nova_env import NovaEnv
from servos import JOINT_ORDER, kinematic_limits

LOW = np.array([kinematic_limits(n)[0] for n in JOINT_ORDER])
HIGH = np.array([kinematic_limits(n)[1] for n in JOINT_ORDER])


def run(env: NovaEnv, actions, seed: int = 0):
    """Eine Episode bis zum Ende. Gibt (Belohnungen, letzte info, umgefallen) zurück."""
    obs, _ = env.reset(seed=seed)
    rewards = []
    while True:
        obs, reward, terminated, truncated, info = env.step(actions(obs))
        rewards.append(reward)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert np.all(LOW - 1e-9 <= env.policy.angles) and np.all(env.policy.angles <= HIGH + 1e-9)
        if terminated or truncated:
            return rewards, info, terminated


@pytest.mark.parametrize("mode", ["residual", "full"])
def test_gymnasium_checker_is_happy(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(NovaEnv(mode=mode), skip_render_check=True)


def test_spaces():
    env = NovaEnv()
    assert env.observation_space.shape == (34,)
    assert env.action_space.shape == (12,)
    assert env.action_space.low.min() == -1.0 and env.action_space.high.max() == 1.0


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        NovaEnv(mode="fliegen")


def test_same_seed_gives_the_same_episode():
    def first_rewards():
        env = NovaEnv()
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        return [env.step(rng.uniform(-1, 1, 12))[1] for _ in range(30)]
    assert first_rewards() == first_rewards()


def test_zero_correction_trots_through_the_whole_episode():
    env = NovaEnv(randomize=False, pushes=False)
    rewards, info, fallen = run(env, lambda obs: np.zeros(12))
    assert not fallen
    assert len(rewards) == round(config.EPISODE_SECONDS * 50)
    assert env.walked > 0.7 * env.commanded > 0  # ohne Netz wie in Phase 1 gemessen: ca. 90 %


@pytest.mark.parametrize("mode", ["residual", "full"])
def test_random_network_runs_through(mode):
    rng = np.random.default_rng(1)
    rewards, info, fallen = run(NovaEnv(mode=mode), lambda obs: rng.uniform(-1, 1, 12), seed=1)
    assert len(rewards) > 0


def test_falling_ends_the_episode_with_a_penalty():
    env = NovaEnv(randomize=False, pushes=False)
    env.reset(seed=0)
    env.sim.push((0.0, 150.0, 0.0), 0.3)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        if terminated:
            break
    assert terminated and not truncated
    assert info["terms"]["umfallen"] == 1.0
    assert reward < config.REWARD["residual"]["umfallen"] + 2.0


def test_reward_is_the_weighted_sum_of_its_terms():
    env = NovaEnv()
    env.reset(seed=5)
    _, reward, _, _, info = env.step(np.full(12, 0.3))
    weights = config.REWARD["residual"]
    assert set(weights) <= set(info["terms"])
    assert reward == pytest.approx(sum(w * info["terms"][k] for k, w in weights.items()))


def test_correction_costs_reward():
    def correction_term(action):
        env = NovaEnv(randomize=False, pushes=False)
        env.reset(seed=0)
        return env.step(action)[4]["terms"]["korrektur"]
    assert correction_term(np.zeros(12)) == 0.0
    assert correction_term(np.full(12, 0.5)) == pytest.approx(0.25)


def test_command_changes_every_2_to_4_seconds():
    env = NovaEnv(randomize=False, pushes=False)
    env.reset(seed=2)
    changes, last = [], (env.policy.target_speed, env.policy.target_yaw_rate)
    for i in range(500):
        env.step(np.zeros(12))
        now = (env.policy.target_speed, env.policy.target_yaw_rate)
        if now != last:
            changes.append((i + 1) / 50)
            last = now
        assert config.COMMAND_SPEED[0] <= now[0] <= config.COMMAND_SPEED[1]
    gaps = np.diff([0.0] + changes)
    assert len(changes) >= 2
    assert np.all((gaps >= 2.0 - 1e-9) & (gaps <= 4.0 + 1e-9))


def test_turning_commands_only_when_switched_on():
    straight = NovaEnv(randomize=False, pushes=False)
    straight.reset(seed=4)
    assert straight.policy.target_yaw_rate == 0.0
    turning = NovaEnv(randomize=False, pushes=False, max_yaw_rate=20.0)
    yaw = []
    for seed in range(10):
        turning.reset(seed=seed)
        yaw.append(turning.policy.target_yaw_rate)
    assert any(y != 0.0 for y in yaw) and all(abs(y) <= 20.0 for y in yaw)


def test_pushes_without_other_randomization():
    env = NovaEnv(randomize=False, pushes=True)
    env.reset(seed=0)
    assert env.randomization.pushes
    assert env.randomization.torso_mass == 1.0 and env.randomization.imu.delay == 0.0


def test_full_mode_rewards_imitating_the_trot():
    env = NovaEnv(mode="full", randomize=False, pushes=False)
    env.reset(seed=0)
    info = env.step(np.zeros(12))[4]
    assert 0.0 < info["terms"]["nachahmung"] <= 1.0
    assert config.REWARD["full"]["nachahmung"] > 0 and config.REWARD["full"]["korrektur"] == 0.0
