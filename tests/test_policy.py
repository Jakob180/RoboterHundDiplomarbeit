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


def test_side_correction_can_be_made_wider():
    m = policy.ResidualMode(foot_scale=(20.0, 40.0, 15.0))
    for _ in range(60):
        m.act(leg_action("RR", y=1.0), DT)
    assert m.correction[LEGS.index("RR") * 3 + 1] == pytest.approx(40.0)
    assert policy.ResidualMode().foot_scale == policy.FOOT_SCALE


def test_unreachable_foot_falls_back_to_the_trot_for_that_leg_only():
    m = policy.ResidualMode(foot_scale=(500.0, 10.0, 15.0))
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


# ---------------------------------------------------------------------------
# Einstellungen und fertiges Netz (ONNX)
# ---------------------------------------------------------------------------

def test_control_rate_is_the_same_as_in_the_simulation():
    import sim
    assert policy.CONTROL_HZ == sim.CONTROL_HZ


@pytest.mark.parametrize("mode, name", [(policy.ResidualMode, "residual"), (policy.FullMode, "full")])
def test_settings_describe_everything_the_observation_needs(mode, name):
    import json
    s = policy.settings(mode())
    assert s["mode"] == name
    assert (s["n_obs"], s["n_actions"]) == (34, 12)
    assert s["joint_order"] == JOINT_ORDER
    assert s["scales"]["gyro"] == policy.GYRO_SCALE
    assert s["scales"]["foot"] == list(policy.FOOT_SCALE)
    assert s["trot"] == {"period": 0.5, "step_height": 25.0, "duty": 0.6}
    assert json.loads(json.dumps(s)) == s


def test_settings_keep_the_foot_scale_of_the_network():
    wide = policy.ResidualMode(foot_scale=(20.0, 40.0, 15.0))
    assert policy.settings(wide)["scales"]["foot"] == [20.0, 40.0, 15.0]


def test_mode_is_rebuilt_from_the_settings_of_a_network():
    wide = policy.mode_from_settings(policy.settings(policy.ResidualMode(foot_scale=(20.0, 40.0, 15.0))))
    assert isinstance(wide, policy.ResidualMode) and wide.foot_scale == (20.0, 40.0, 15.0)
    assert isinstance(policy.mode_from_settings(policy.settings(policy.FullMode())), policy.FullMode)


def test_network_with_its_own_foot_scale_is_accepted():
    policy.check_settings(policy.settings(policy.ResidualMode(foot_scale=(20.0, 40.0, 15.0))))


def write_linear_network(path, weights, bias, meta) -> None:
    """Kleines Test-Netz: Aktion = obs · weights + bias, mit .json daneben."""
    import json
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["xw"]), helper.make_node("Add", ["xw", "b"], ["action"])],
        "test",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", 34])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", 12])],
        [numpy_helper.from_array(weights.astype(np.float32), "W"), numpy_helper.from_array(bias.astype(np.float32), "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def test_onnx_network_computes_the_actions(tmp_path):
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    w, b = rng.normal(size=(34, 12)), rng.normal(size=12)
    write_linear_network(tmp_path / "netz.onnx", w, b, policy.settings(policy.ResidualMode()))
    net = policy.OnnxNetwork(tmp_path / "netz.onnx")
    assert net.mode == "residual"
    obs = rng.normal(size=34).astype(np.float32)
    assert net(obs) == pytest.approx(obs @ w + b, abs=1e-4)


def test_onnx_network_refuses_other_settings(tmp_path):
    pytest.importorskip("onnxruntime")
    meta = policy.settings(policy.ResidualMode())
    meta["scales"]["gyro"] = 2.0  # z. B. mit einer älteren policy.py trainiert
    write_linear_network(tmp_path / "netz.onnx", np.zeros((34, 12)), np.zeros(12), meta)
    with pytest.raises(ValueError, match="scales.gyro"):
        policy.OnnxNetwork(tmp_path / "netz.onnx")


@pytest.mark.parametrize("bad", [(20.0, -10.0, 15.0), (20.0, 40.0), (0.0, 10.0, 15.0)])
def test_foot_scale_needs_three_positive_values(bad):
    with pytest.raises(ValueError, match="foot_scale"):
        policy.ResidualMode(foot_scale=bad)


def test_settings_contain_the_leg_geometry():
    import gait
    import kinematics
    g = policy.settings(policy.ResidualMode())["geometry"]
    assert g == {
        "hip": kinematics.HIP_LENGTH, "upper": kinematics.UPPER_LENGTH, "lower": kinematics.LOWER_LENGTH,
        "body_length": kinematics.BODY_LENGTH, "body_width": kinematics.BODY_WIDTH,
        "stand_height": gait.STAND.body.height,
    }


def test_network_trained_on_other_legs_is_refused():
    found = policy.settings(policy.ResidualMode())
    found["geometry"]["upper"] = 107.5  # z. B. mit den alten Spot-Micro-Maßen trainiert
    with pytest.raises(ValueError, match="geometry.upper"):
        policy.check_settings(found)


def test_networks_from_before_the_geometry_check_are_refused():
    old = policy.settings(policy.ResidualMode())
    del old["geometry"]
    old["version"] = 1
    with pytest.raises(ValueError, match="version"):
        policy.check_settings(old)
