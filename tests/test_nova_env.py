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


def test_scenario_can_be_given_for_fair_comparisons():
    import randomization as rnd
    r = rnd.Randomization(floor_friction=0.5)
    env = NovaEnv()
    env.reset(seed=0, options={"randomization": r, "command": (150.0, 10.0)})
    assert env.randomization is r
    floor = env.sim.model.geom("floor").id
    assert env.sim.model.geom_friction[floor, 0] == 0.5
    for _ in range(300):  # 6 s: der Befehl bleibt fest
        env.step(np.zeros(12))
        assert (env.policy.target_speed, env.policy.target_yaw_rate) == (150.0, 10.0)


def test_given_scenario_gives_the_same_episode_every_time():
    import randomization as rnd
    r = rnd.sample(np.random.default_rng(7))

    def rewards():
        env = NovaEnv()
        env.reset(seed=1, options={"randomization": r, "command": (120.0, 0.0)})
        return [env.step(np.zeros(12))[1] for _ in range(50)]
    assert rewards() == rewards()


def test_episode_end_reports_the_walked_share():
    env = NovaEnv(randomize=False, pushes=False, episode_seconds=2.0)
    env.reset(seed=0, options={"command": (100.0, 0.0)})
    for _ in range(100):
        *_, terminated, truncated, info = env.step(np.zeros(12))
    assert truncated
    # eigener Schlüssel "nova": "episode" belegt Stable-Baselines3 selbst
    assert info["nova"]["strecke"] == pytest.approx(env.walked / env.commanded)
    assert info["nova"]["umgefallen"] == 0.0


# ---------------------------------------------------------------------------
# Kurs halten
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("actual, planned, error", [
    (0.0, 0.0, 0.0), (15.0, 5.0, 10.0), (-30.0, 0.0, -30.0), (179.0, -179.0, -2.0), (10.0, 370.0, 0.0),
])
def test_course_error_in_degrees(actual, planned, error):
    from nova_env import course_error
    assert course_error(actual, planned) == pytest.approx(error)


def turn_torso(env: NovaEnv, degrees: float) -> None:
    """Rumpf um die Hochachse verdrehen (wie nach einem Abdriften)."""
    import math
    import mujoco
    half = math.radians(degrees) / 2
    env.sim.data.qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
    mujoco.mj_forward(env.sim.model, env.sim.data)


def test_staying_on_course_scores_full():
    env = NovaEnv(randomize=False, pushes=False)
    env.reset(seed=0, options={"command": (100.0, 0.0)})
    assert env.step(np.zeros(12))[4]["terms"]["kurs"] > 0.95


def test_drifting_off_course_costs_reward():
    env = NovaEnv(randomize=False, pushes=False)
    env.reset(seed=0, options={"command": (100.0, 0.0)})
    turn_torso(env, 20.0)
    assert env.step(np.zeros(12))[4]["terms"]["kurs"] < 0.1   # 20° daneben, Breite 10°


def test_commanded_turn_is_the_course():
    env = NovaEnv(randomize=False, pushes=False)
    env.reset(seed=0, options={"command": (100.0, 20.0)})
    for _ in range(150):
        env.step(np.zeros(12))
    turn_torso(env, env.policy.trot.heading)   # genau so weit gedreht wie befohlen
    assert env.step(np.zeros(12))[4]["terms"]["kurs"] > 0.9


def test_course_is_rewarded_in_both_stages():
    assert config.REWARD["residual"]["kurs"] > 0 and config.REWARD["full"]["kurs"] > 0


def test_push_strength_reaches_the_randomization():
    import math
    env = NovaEnv(randomize=True, pushes=True, push_force=25.0, episode_seconds=60.0)
    forces = []
    for seed in range(5):
        env.reset(seed=seed)
        forces += [math.hypot(p.force[0], p.force[1]) for p in env.randomization.pushes]
    assert max(forces) <= 25.0 and max(forces) > 15.0


def test_foot_scale_reaches_the_policy():
    assert NovaEnv(foot_scale=(20.0, 40.0, 15.0)).policy.foot_scale == (20.0, 40.0, 15.0)


def test_push_style_reaches_the_randomization():
    import math
    from randomization import PUSH_STYLES
    style = PUSH_STYLES["stark-seitlich"]
    for randomize in (True, False):
        env = NovaEnv(randomize=randomize, pushes=True, push_style=style, episode_seconds=30.0)
        env.reset(seed=0)
        forces = [math.hypot(p.force[0], p.force[1]) for p in env.randomization.pushes]
        assert forces and min(forces) >= style.force[0]


def test_stage_2_lets_the_legs_bump_into_each_other():
    full, residual = NovaEnv(mode="full"), NovaEnv(mode="residual")
    assert full.self_collision and not residual.self_collision
    full.reset(seed=0)
    leg = full.sim.model.geom("FL_foot").id
    assert full.sim.model.geom_conaffinity[leg] == 1


def test_reward_weights_belong_to_one_environment():
    env = NovaEnv(mode="full")
    env.set_reward_weight("nachahmung", 0.2)
    assert env.weights["nachahmung"] == 0.2
    assert config.REWARD["full"]["nachahmung"] == 1.0


def mean_term(env, name, actions, seconds=6.0, **reset):
    env.reset(seed=0, **reset)
    values = []
    for _ in range(round(seconds * 50)):
        _, _, fallen, _, info = env.step(actions)
        values.append(info["terms"][name])
        if fallen:
            break
    return float(np.mean(values))


def test_feet_slipping_on_a_slippery_floor_cost_reward():
    import randomization as rnd
    env = NovaEnv(mode="residual", randomize=False, pushes=False)
    grip = mean_term(env, "fuss_rutschen", np.zeros(12), options={"command": (150.0, 0.0)})
    slippery = mean_term(env, "fuss_rutschen", np.zeros(12),
                         options={"command": (150.0, 0.0), "randomization": rnd.Randomization(floor_friction=0.1)})
    assert slippery > 1.5 * grip > 0  # gemessen 1,8-mal so viel


def test_dragging_feet_cost_reward():
    standing = mean_term(NovaEnv(mode="full", randomize=False, pushes=False), "fusshoehe", np.zeros(12),
                         options={"command": (150.0, 0.0)})
    trotting = mean_term(NovaEnv(mode="residual", randomize=False, pushes=False), "fusshoehe", np.zeros(12),
                         options={"command": (150.0, 0.0)})
    assert standing > 0.3 and trotting < standing / 2


def test_stage_2_pays_for_clean_feet_stage_1_stays_as_it_was():
    assert config.REWARD["full"]["fuss_rutschen"] < 0 and config.REWARD["full"]["fusshoehe"] < 0
    assert "fuss_rutschen" not in config.REWARD["residual"] and "fusshoehe" not in config.REWARD["residual"]
