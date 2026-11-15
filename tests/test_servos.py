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
