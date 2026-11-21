"""Kalibrierung Kinematik-Winkel <-> Servo-Winkel und die Gelenkgrenzen."""

import pytest

import gait
import servos
from servos import JOINT_ORDER, JOINTS


def test_kinematic_zero_is_the_calibrated_servo_angle():
    assert servos.to_servo_angle("FL_upper", 0.0) == JOINTS["FL_upper"].zero


@pytest.mark.parametrize("name", JOINT_ORDER)
def test_conversion_there_and_back_gives_the_same_angle(name):
    assert servos.to_kinematic_angle(name, servos.to_servo_angle(name, -37.5)) == pytest.approx(-37.5)


@pytest.mark.parametrize("name", JOINT_ORDER)
def test_kinematic_limits_are_the_servo_limits_converted(name):
    low, high = servos.kinematic_limits(name)
    assert low < high
    assert servos.to_servo_angle(name, low) == pytest.approx(JOINTS[name].min_angle)
    assert servos.to_servo_angle(name, high) == pytest.approx(JOINTS[name].max_angle)


@pytest.mark.parametrize("name", JOINT_ORDER)
def test_limits_fit_into_the_180_degrees_of_the_servo(name):
    joint = JOINTS[name]
    assert 0.0 <= joint.min_angle <= joint.neutral <= joint.max_angle <= 180.0


def test_neutral_position_is_roughly_the_standing_pose():
    stand = gait.body_ik(gait.STAND.feet, gait.STAND.body)
    for name in JOINT_ORDER:
        standing = getattr(stand[name[:2]], name[3:])
        assert servos.to_kinematic_angle(name, JOINTS[name].neutral) == pytest.approx(standing, abs=5.0)


def test_whole_demo_sequence_stays_inside_the_joint_limits():
    for frame in gait.demo_frames():
        for name in JOINT_ORDER:
            low, high = servos.kinematic_limits(name)
            assert low <= getattr(frame.angles[name[:2]], name[3:]) <= high, (frame.text, name)


def test_angle_step_is_one_pca9685_tick():
    # 50 Hz: 20 000 µs / 4096 Stufen = 4,88 µs; 2000 µs sind 180°
    assert servos.ANGLE_STEP == pytest.approx(180.0 / 409.6)


# ---------------------------------------------------------------------------
# Kalibrier-Datei (aus src/calibrate.py)
# ---------------------------------------------------------------------------

def test_tests_use_the_placeholder_values():
    assert not servos.CALIBRATION_FILE.exists()
    assert servos.JOINTS["FL_upper"].zero == 140


def test_calibration_file_replaces_the_placeholders(tmp_path):
    import json
    path = tmp_path / "kalibrierung.json"
    path.write_text(json.dumps({"joints": {
        "FR_upper": {"min": 35.0, "max": 160.0, "neutral": 96.0, "zero": 146.0, "inverted": False},
    }}), encoding="utf-8")
    joints = servos.load_calibration(path, servos.JOINTS)
    assert joints["FR_upper"] == servos.Joint(4, 35.0, 160.0, 96.0, 146.0, False)  # Kanal bleibt
    assert joints["FL_upper"] == servos.JOINTS["FL_upper"]                         # Rest unverändert


@pytest.mark.parametrize("bad", [
    {"NOPE_upper": {"min": 30, "max": 150, "neutral": 90, "zero": 140, "inverted": False}},
    {"FL_upper": {"min": 150, "max": 30, "neutral": 90, "zero": 140, "inverted": False}},
    {"FL_upper": {"min": 30, "max": 190, "neutral": 90, "zero": 140, "inverted": False}},
    {"FL_upper": {"min": 30, "max": 150, "neutral": 20, "zero": 140, "inverted": False}},
])
def test_broken_calibration_is_refused(tmp_path, bad):
    import json
    path = tmp_path / "kalibrierung.json"
    path.write_text(json.dumps({"joints": bad}), encoding="utf-8")
    with pytest.raises(ValueError):
        servos.load_calibration(path, servos.JOINTS)


def test_raw_angle_for_calibration_stays_inside_the_servo_range():
    s = servos.Servos()
    assert s.set_raw(1, 200.0) == servos.RAW_RANGE[1]
    assert s.set_raw(1, -20.0) == servos.RAW_RANGE[0]
    assert s.set_raw(1, 95.0) == 95.0
    s.release(1)
    assert s._servo[1].angle is None
