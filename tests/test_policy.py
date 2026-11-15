"""Schnittstelle zum Netz: Beobachtung und sichere Gelenkwinkel, Stufe 1 und 2."""

import math

import numpy as np
import pytest

import gait
import policy
import servos
from kinematics import LEGS, body_ik, foot_ik
from servos import JOINT_ORDER

DT = 0.02
LEVEL = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))  # Lagesensor: gerade, ruhig
LOW = np.array([servos.kinematic_limits(n)[0] for n in JOINT_ORDER])
HIGH = np.array([servos.kinematic_limits(n)[1] for n in JOINT_ORDER])


def as_array(angles) -> np.ndarray:
    return np.array([getattr(angles[n[:2]], n[3:]) for n in JOINT_ORDER])


def leg_action(leg: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    a = np.zeros(12)
    i = LEGS.index(leg)
    a[3 * i:3 * i + 3] = (x, y, z)
    return a


MODES = [policy.ResidualMode, policy.FullMode]


# ---------------------------------------------------------------------------
# Beobachtung
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_observation_has_34_values(mode):
    obs = mode().observe(*LEVEL)
    assert obs.shape == (34,)
    assert obs.dtype == np.float32


def test_observation_layout_stage_1():
    m = policy.ResidualMode()
    m.set_command(100.0, 0.0)
    m.act(np.zeros(12), DT)
    obs = m.observe((0.0, 0.1, 0.99), (0.4, -0.8, 0.0))
    assert obs[0:3] == pytest.approx((0.0, 0.1, 0.99))
    assert obs[3:6] == pytest.approx((0.1, -0.2, 0.0))          # ÷ 4 rad/s
    assert obs[6:8] == pytest.approx((m.speed / 200.0, 0.0))    # ÷ Höchstwert
    assert obs[8:10] == pytest.approx((math.sin(2 * math.pi * m.trot.phase), math.cos(2 * math.pi * m.trot.phase)))
    assert obs[10:22] == pytest.approx(np.zeros(12))            # letzte Aktion
    stand = as_array(body_ik(gait.STAND.feet, gait.STAND.body))
    assert obs[22:34] == pytest.approx((m.angles - stand) / 45.0, abs=1e-6)


def test_observation_stage_2_has_the_last_two_imu_readings():
    m = policy.FullMode()
    m.observe((0.0, 0.0, 1.0), (0.4, 0.0, 0.0))
    m.observe((0.1, 0.0, 0.99), (0.0, 0.8, 0.0))
    obs = m.observe((0.2, 0.0, 0.98), (0.0, 0.0, 1.2))
    assert obs[22:28] == pytest.approx((0.1, 0.0, 0.99, 0.0, 0.2, 0.0))   # vor 1 Schritt
    assert obs[28:34] == pytest.approx((0.0, 0.0, 1.0, 0.1, 0.0, 0.0))    # vor 2 Schritten


def test_observation_is_clipped():
    obs = policy.ResidualMode().observe((0.0, 0.0, 1.0), (100.0, 0.0, 0.0))
    assert obs[3] == policy.OBS_CLIP


# ---------------------------------------------------------------------------
# Befehl
# ---------------------------------------------------------------------------

def test_command_is_approached_smoothly():
    m = policy.ResidualMode()
    m.set_command(200.0, 20.0)
    m.act(np.zeros(12), DT)
    assert m.speed == pytest.approx(policy.SPEED_RAMP * DT)
    assert m.yaw_rate == pytest.approx(policy.YAW_RAMP * DT)
    for _ in range(100):
        m.act(np.zeros(12), DT)
    assert (m.speed, m.yaw_rate) == pytest.approx((200.0, 20.0))


def test_command_is_limited():
    m = policy.ResidualMode()
    m.set_command(500.0, -100.0)
    assert (m.target_speed, m.target_yaw_rate) == pytest.approx((policy.MAX_SPEED, -policy.MAX_YAW_RATE))


# ---------------------------------------------------------------------------
# Stufe 1: Trab + Korrektur
# ---------------------------------------------------------------------------

def test_zero_correction_is_exactly_the_classic_trot():
    m = policy.ResidualMode()
    trot = gait.Trot()
    m.set_command(150.0, 10.0)
    for i in range(150):
        got = m.act(np.zeros(12), DT)
        speed = min(150.0, policy.SPEED_RAMP * DT * (i + 1))
        yaw_rate = min(10.0, policy.YAW_RAMP * DT * (i + 1))
        pose = trot.update(DT, speed, yaw_rate)
        assert as_array(got) == pytest.approx(as_array(body_ik(pose.feet, pose.body)), abs=1e-9), i


def test_correction_moves_the_foot():
    m = policy.ResidualMode()
    got = m.act(leg_action("FL", x=1.0, z=-1.0), DT)
    pose = gait.Trot().update(DT, 0.0)
    share = 1.0 - policy.SMOOTHING
    fx, fy, fz = pose.feet["FL"]
    sx, _, sz = policy.FOOT_SCALE
    assert got["FL"] == pytest.approx(foot_ik("FL", (fx + share * sx, fy, fz - share * sz), pose.body))
    assert got["FR"] == pytest.approx(foot_ik("FR", pose.feet["FR"], pose.body))


def test_held_correction_reaches_its_full_size():
    m = policy.ResidualMode()
    for _ in range(60):
        m.act(leg_action("RR", y=1.0), DT)
    assert m.correction[LEGS.index("RR") * 3 + 1] == pytest.approx(policy.FOOT_SCALE[1])


def test_unreachable_foot_falls_back_to_the_trot_for_that_leg_only(monkeypatch):
    monkeypatch.setattr(policy, "FOOT_SCALE", (500.0, 10.0, 15.0))
    m = policy.ResidualMode()
    got = m.act(leg_action("RL", x=1.0) + leg_action("FR", y=1.0), DT)
    pose = gait.Trot().update(DT, 0.0)
    assert got["RL"] == pytest.approx(foot_ik("RL", pose.feet["RL"], pose.body))
    fx, fy, fz = pose.feet["FR"]
    assert got["FR"] == pytest.approx(foot_ik("FR", (fx, fy + (1 - policy.SMOOTHING) * 10.0, fz), pose.body))


# ---------------------------------------------------------------------------
# Stufe 2: Netz gibt die Gelenkwinkel vor
# ---------------------------------------------------------------------------

def test_stage_2_zero_output_is_the_standing_pose():
    m = policy.FullMode()
    got = m.act(np.zeros(12), DT)
    assert as_array(got) == pytest.approx(as_array(body_ik(gait.STAND.feet, gait.STAND.body)))


def test_stage_2_joints_move_at_most_400_degrees_per_second():
    m = policy.FullMode()
    start = m.angles.copy()
    a = np.zeros(12)
    a[JOINT_ORDER.index("FL_upper")] = 1.0
    m.act(a, DT)
    assert m.angles[JOINT_ORDER.index("FL_upper")] - start[JOINT_ORDER.index("FL_upper")] == pytest.approx(8.0)
    for _ in range(10):
        m.act(a, DT)
    target = start[JOINT_ORDER.index("FL_upper")] + policy.JOINT_SCALE["upper"]
    assert m.angles[JOINT_ORDER.index("FL_upper")] == pytest.approx(min(target, HIGH[JOINT_ORDER.index("FL_upper")]))


def test_stage_2_reference_is_the_trot():
    m = policy.FullMode()
    trot = gait.Trot()
    m.set_command(100.0, 0.0)
    for i in range(30):
        m.act(np.zeros(12), DT)
        pose = trot.update(DT, min(100.0, policy.SPEED_RAMP * DT * (i + 1)))
    assert m.reference == pytest.approx(as_array(body_ik(pose.feet, pose.body)))


# ---------------------------------------------------------------------------
# Sicherheit: Grenzen und unsinnige Ausgaben
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_any_output_stays_inside_the_joint_limits(mode):
    rng = np.random.default_rng(0)
    m = mode()
    m.set_command(200.0, 28.0)
    for i in range(300):
        a = rng.uniform(-1, 1, 12) if i % 3 else rng.choice([-50.0, 50.0], 12)
        got = as_array(m.act(a, DT))
        assert np.all(LOW - 1e-9 <= got) and np.all(got <= HIGH + 1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_nan_output_is_treated_as_zero(mode):
    a, b = mode(), mode()
    bad = np.zeros(12)
    bad[5] = np.nan
    assert as_array(a.act(bad, DT)) == pytest.approx(as_array(b.act(np.zeros(12), DT)))
    assert np.all(a.last_action == 0.0)


@pytest.mark.parametrize("mode", MODES)
def test_last_action_is_the_clipped_output(mode):
    m = mode()
    m.act(np.full(12, 3.0), DT)
    assert np.all(m.last_action == 1.0)


@pytest.mark.parametrize("mode", MODES)
def test_reset_starts_over(mode):
    m = mode()
    m.set_command(150.0, 0.0)
    for _ in range(20):
        m.act(np.full(12, 0.5), DT)
    m.reset()
    fresh = mode()
    assert m.observe(*LEVEL) == pytest.approx(fresh.observe(*LEVEL))
    assert (m.speed, m.target_speed, m.trot.phase) == (0.0, 0.0, 0.0)
