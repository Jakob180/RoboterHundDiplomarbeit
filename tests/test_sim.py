"""Servo-Modell und Simulation."""

import math

import mujoco
import numpy as np
import pytest

import gait
import servos
import sim
from kinematics import LegAngles, body_ik
from servos import JOINT_ORDER, JOINTS

P = sim.ServoParams()
STAND = body_ik(gait.STAND.feet, gait.STAND.body)


def torque(error_deg: float, speed_deg: float = 0.0, params: sim.ServoParams = P) -> float:
    return float(sim.servo_torque(np.radians([error_deg]), np.radians([speed_deg]), params)[0])


# ---------------------------------------------------------------------------
# Servo-Modell als reine Rechnung
# ---------------------------------------------------------------------------

def test_no_error_and_no_motion_gives_no_torque():
    assert torque(0.0) == 0.0


def test_inside_the_dead_band_the_servo_does_nothing():
    assert torque(0.2) == 0.0
    assert torque(-0.2) == 0.0


def test_outside_the_dead_band_the_torque_grows_with_the_error():
    expected = P.kp * math.radians(1.0 - 0.27)
    assert torque(1.0) == pytest.approx(expected)
    assert torque(-1.0) == pytest.approx(-expected)


def test_large_error_gives_at_most_the_stall_torque():
    assert torque(30.0) == pytest.approx(P.stall_torque)
    assert torque(-30.0) == pytest.approx(-P.stall_torque)


def test_torque_drops_with_speed_in_the_direction_of_motion():
    half = math.degrees(P.max_speed) / 2
    assert torque(30.0, half) == pytest.approx(P.stall_torque / 2)
    assert torque(-30.0, -half) == pytest.approx(-P.stall_torque / 2)
    assert torque(30.0, math.degrees(P.max_speed)) == pytest.approx(0.0, abs=1e-12)


def test_braking_against_the_motion_has_the_full_stall_torque():
    assert torque(-30.0, 300.0) == pytest.approx(-P.stall_torque)


def test_motion_without_error_is_damped():
    assert torque(0.0, 10.0) == pytest.approx(-P.kd * math.radians(10.0))


def test_commanded_angles_are_clamped_and_nan_goes_to_neutral():
    wanted = np.zeros(12)
    wanted[JOINT_ORDER.index("FL_upper")] = 200.0
    wanted[JOINT_ORDER.index("FL_lower")] = float("nan")
    got = sim.command_angles(wanted)
    step = servos.ANGLE_STEP
    assert got[JOINT_ORDER.index("FL_upper")] == pytest.approx(servos.kinematic_limits("FL_upper")[1], abs=step / 2)
    neutral = servos.to_kinematic_angle("FL_lower", JOINTS["FL_lower"].neutral)
    assert got[JOINT_ORDER.index("FL_lower")] == pytest.approx(neutral, abs=step / 2)


def test_commanded_angles_are_rounded_to_pca9685_steps():
    wanted = sim.NovaSim.joint_angles(STAND) + 0.1  # sicher innerhalb der Grenzen
    got = sim.command_angles(wanted)
    for name, angle, want in zip(JOINT_ORDER, got, wanted):
        steps = servos.to_servo_angle(name, angle) / servos.ANGLE_STEP
        assert steps == pytest.approx(round(steps), abs=1e-9)
        assert angle == pytest.approx(want, abs=servos.ANGLE_STEP / 2)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def in_the_air(params: sim.ServoParams = P) -> sim.NovaSim:
    """Hund ohne Schwerkraft, damit nur der Servo zählt."""
    s = sim.NovaSim(params)
    s.model.opt.gravity[:] = 0.0
    s.reset(STAND, 300.0)
    return s


def joint_deg(s: sim.NovaSim, name: str) -> float:
    return math.degrees(s.data.qpos[s.model.joint(name).qposadr[0]])


def with_angle(name: str, angle: float) -> dict[str, LegAngles]:
    angles = dict(STAND)
    angles[name[:2]] = angles[name[:2]]._replace(**{name[3:]: angle})
    return angles


def test_standing_dog_stays_up_and_level():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    for _ in range(50):
        s.step(STAND)
    t = s.torso()
    assert t.z == pytest.approx(gait.STAND.body.height, abs=10.0)
    assert abs(t.roll) < 1.0 and abs(t.pitch) < 1.0


def test_servo_reaches_its_target():
    s = in_the_air()
    target = STAND["FL"].upper + 20.0
    for _ in range(25):
        s.step(with_angle("FL_upper", target))
    # Rest: Totzone 0,27° + PCA9685-Stufe bis 0,22°
    assert joint_deg(s, "FL_upper") == pytest.approx(target, abs=0.6)


def test_joint_speed_is_limited_by_the_servo():
    s = in_the_air()
    adr = s.model.joint("FL_lower").dofadr[0]
    fastest = 0.0
    for _ in range(15):
        s.step(with_angle("FL_lower", 140.0))
        fastest = max(fastest, abs(s.data.qvel[adr]))
    assert 0.9 * P.max_speed < fastest <= P.max_speed


def test_dead_time_delays_the_reaction():
    def moved_after_one_step(dead_time: float) -> float:
        s = in_the_air(sim.ServoParams(dead_time=dead_time))
        before = joint_deg(s, "FL_upper")
        s.step(with_angle("FL_upper", before + 20.0))
        return abs(joint_deg(s, "FL_upper") - before)

    assert moved_after_one_step(0.040) < 0.1
    assert moved_after_one_step(0.004) > 1.0


def test_calibration_error_shifts_the_joint():
    offsets = np.zeros(12)
    offsets[JOINT_ORDER.index("RR_lower")] = math.radians(3.0)
    s = in_the_air(sim.ServoParams(offsets=offsets))
    for _ in range(25):
        s.step(STAND)
    assert joint_deg(s, "RR_lower") == pytest.approx(STAND["RR"].lower + 3.0, abs=0.6)


def test_step_can_take_a_different_control_period():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    s.step(STAND, period=0.018)
    assert s.data.time == pytest.approx(0.018)


def test_imu_feels_gravity_when_standing():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    for _ in range(25):
        s.step(STAND)
    gyro, accel = s.imu()
    assert gyro == pytest.approx((0.0, 0.0, 0.0), abs=0.01)
    assert accel == pytest.approx((0.0, 0.0, 9.81), abs=0.2)


def test_distance_is_none_on_open_ground():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    assert s.distance() is None


def test_distance_is_in_mm():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    tilt = math.radians(30.0)
    s.data.qpos[2] = 0.2
    s.data.qpos[3:7] = (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0)
    mujoco.mj_forward(s.model, s.data)
    assert s.distance() == pytest.approx(1000 * s.data.sensor("range").data[0])


def test_push_moves_the_dog_and_stops_after_its_duration():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    s.push((0.0, 10.0, 0.0), 0.1)
    for _ in range(5):
        s.step(STAND)
    assert s.torso().y > 1.0
    assert not s.data.xfrc_applied.any()


def test_classic_trot_still_walks():
    frames = gait.demo_frames(sim.CONTROL_HZ)
    s = sim.NovaSim()
    s.reset(frames[0].angles, frames[0].pose.body.height)
    for frame in frames:
        s.step(frame.angles)
        t = s.torso()
        assert t.z > 100 and abs(t.roll) < 10 and abs(t.pitch - frame.pose.body.pitch) < 10, frame.text
    # Ausgangswert mit den Platzhaltern: 90 % (Servos geben unter 2,6 kg etwas nach)
    assert s.torso().x / frames[-1].distance > 0.8


@pytest.mark.parametrize("yaw_rate", [20.0, -20.0])
def test_turning_trot_turns_the_dog(yaw_rate):
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    trot = gait.Trot()
    for i in range(250):  # 1 s anlaufen, dann 4 s drehen
        v, w = (100.0 * i / 50, 0.0) if i < 50 else (100.0, yaw_rate)
        pose = trot.update(1.0 / sim.CONTROL_HZ, v, w)
        s.step(body_ik(pose.feet, pose.body))
    assert trot.heading == pytest.approx(4 * yaw_rate)
    # gemessen: 83° bzw. -84° bei geplanten ±80°
    assert s.torso().yaw == pytest.approx(trot.heading, rel=0.3)


def test_imu_can_be_read_with_a_delay():
    # Schritte von 2 ms (ein Physikschritt), dabei schubsen, damit sich die Werte ändern
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    s.push((0.0, 15.0, 0.0), 0.05)
    readings = []
    for _ in range(12):
        s.step(STAND, period=0.002)
        readings.append(s.imu())
    assert s.imu(delay=0.0) == readings[-1]
    assert s.imu(delay=0.006) == readings[-4]
    assert s.imu(delay=0.006) != readings[-1]


def test_imu_delay_longer_than_the_record_gives_the_oldest_value():
    s = sim.NovaSim()
    s.reset(STAND, gait.STAND.body.height)
    first = s.imu()
    s.step(STAND, period=0.002)
    assert s.imu(delay=1.0) == first
