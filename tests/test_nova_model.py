"""Das MuJoCo-Modell: Massen, Reibung, Sensoren, Gelenkgrenzen, Antriebe."""

import math

import mujoco
import numpy as np
import pytest

import gait
import nova_model
import servos
from kinematics import HIP_LENGTH, body_ik, leg_ik
from servos import JOINT_ORDER


def load(**options):
    model = mujoco.MjModel.from_xml_string(nova_model.build_mjcf(**options))
    return model, mujoco.MjData(model)


def pose(model, data, angles, height: float) -> None:
    """Hund in eine Pose stellen (Kinematik-Winkel je Bein, Höhe der Körpermitte in mm)."""
    data.qpos[2] = height / 1000
    for name in JOINT_ORDER:
        data.qpos[model.joint(name).qposadr[0]] = math.radians(getattr(angles[name[:2]], name[3:]))
    mujoco.mj_forward(model, data)


def stand(model, data, height: float = 169.0) -> None:
    """Stehpose, 1 mm zu tief, damit alle 4 Füße sicher Bodenkontakt haben."""
    pose(model, data, body_ik(gait.STAND.feet, gait.STAND.body), height)


def self_contacts(model, data) -> int:
    floor = model.geom("floor").id
    return sum(1 for c in data.contact[:data.ncon] if floor not in (c.geom1, c.geom2))


def test_total_mass_is_about_2_6_kg():
    model, _ = load()
    assert float(model.body_mass.sum()) == pytest.approx(2.6, abs=0.05)


def test_torso_centre_of_mass_follows_battery_and_powerbank():
    model, _ = load()
    parts = (
        (nova_model.FRAME_MASS, (0.0, 0.0, 0.0)),
        (nova_model.BATTERY_MASS, nova_model.BATTERY_POS),
        (nova_model.POWERBANK_MASS, nova_model.POWERBANK_POS),
    )
    mass = sum(m for m, _ in parts)
    centre = sum(m * np.array(p) for m, p in parts) / mass
    torso = model.body("torso")
    assert torso.mass[0] == pytest.approx(mass)
    assert torso.ipos * 1000 == pytest.approx(centre, abs=1e-3)


def test_servo_data_lies_between_the_datasheet_values_for_4_8_and_8_4_volt():
    assert 2.30 < nova_model.SERVO_TORQUE < 2.70
    assert 400 < math.degrees(nova_model.SERVO_SPEED) < 462


def test_floor_friction_decides_the_grip_of_the_feet():
    model, data = load()
    stand(model, data)
    assert data.ncon == 4
    assert [c.friction[0] for c in data.contact[:4]] == pytest.approx([nova_model.FLOOR_FRICTION] * 4)

    model.geom_friction[model.geom("floor").id, 0] = 0.3
    mujoco.mj_forward(model, data)
    assert [c.friction[0] for c in data.contact[:4]] == pytest.approx([0.3] * 4)


def test_floor_softness_decides_the_contact():
    model, data = load()
    model.geom_solref[model.geom("floor").id] = (0.03, 1.0)
    stand(model, data)
    assert list(data.contact[0].solref) == pytest.approx([0.03, 1.0])


def test_imu_sensors_sit_on_the_torso():
    model, _ = load()
    site = model.site("imu")
    assert site.bodyid[0] == model.body("torso").id
    assert site.pos * 1000 == pytest.approx(nova_model.IMU_POS)
    for name, kind in (("gyro", mujoco.mjtSensor.mjSENS_GYRO), ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        sensor = model.sensor(name)
        assert sensor.type[0] == kind
        assert sensor.objid[0] == site.id


def test_rangefinder_looks_forward():
    # Rumpf 30° mit der Nase nach unten in der Luft: der Strahl trifft den Boden schräg vor dem Hund
    model, data = load()
    tilt = math.radians(30.0)
    data.qpos[2] = 0.2
    data.qpos[3:7] = (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0)
    mujoco.mj_forward(model, data)
    front = nova_model.RANGE_POS[0] / 1000
    expected = (0.2 - front * math.sin(tilt)) / math.sin(tilt)
    assert data.sensor("range").data[0] == pytest.approx(expected, abs=1e-6)


def test_rangefinder_sees_nothing_on_open_ground():
    model, data = load()
    stand(model, data)
    assert data.sensor("range").data[0] == -1.0


@pytest.mark.parametrize("name", JOINT_ORDER)
def test_joint_limits_come_from_the_calibration_in_servos(name):
    model, _ = load()
    assert np.degrees(model.joint(name).range) == pytest.approx(servos.kinematic_limits(name), abs=0.01)


def test_actuators_are_torque_motors_in_joint_order():
    model, data = load()
    assert [model.actuator(i).name for i in range(model.nu)] == JOINT_ORDER
    data.ctrl[:] = 1.5  # N·m
    mujoco.mj_forward(model, data)
    assert list(data.actuator_force) == pytest.approx([1.5] * 12)


def cross_left_feet(model, data) -> None:
    """Vorderes und hinteres linkes Bein so stellen, dass sich die Füße überlappen (Hund in der Luft)."""
    angles = body_ik(gait.STAND.feet, gait.STAND.body)
    angles["FL"] = leg_ik(-88.0, HIP_LENGTH, -150.0)
    angles["RL"] = leg_ik(88.0, HIP_LENGTH, -150.0)
    pose(model, data, angles, 300.0)


def test_legs_pass_through_each_other_by_default():
    model, data = load()
    cross_left_feet(model, data)
    assert self_contacts(model, data) == 0


def test_self_collision_switch_makes_the_legs_collide():
    model, data = load(self_collision=True)
    cross_left_feet(model, data)
    assert self_contacts(model, data) > 0


def test_standing_pose_has_no_self_collisions():
    model, data = load(self_collision=True)
    stand(model, data)
    assert self_contacts(model, data) == 0
    assert data.ncon == 4
