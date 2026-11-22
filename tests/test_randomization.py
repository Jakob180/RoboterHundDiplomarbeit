"""Domain Randomization: gewürfelte Werte für einen Durchlauf (Plan Abschnitt 5.2)."""

import math

import numpy as np
import pytest

import nova_model
import randomization as rnd
import sim


def sample(seed: int, **options) -> rnd.Randomization:
    return rnd.sample(np.random.default_rng(seed), **options)


def test_same_seed_gives_the_same_values():
    a, b = sample(7), sample(7)
    assert (a.torso_mass, a.floor_friction, a.servo.stall_torque, a.pushes) == \
           (b.torso_mass, b.floor_friction, b.servo.stall_torque, b.pushes)
    assert np.array_equal(a.servo.offsets, b.servo.offsets)


def test_different_seeds_give_different_values():
    assert sample(1).floor_friction != sample(2).floor_friction


@pytest.mark.parametrize("seed", range(30))
def test_values_stay_inside_the_ranges_from_the_plan(seed):
    r = sample(seed)
    assert 0.8 <= r.torso_mass <= 1.2
    assert all(abs(v) <= 20.0 for v in r.torso_com)
    assert 0.85 <= r.leg_mass <= 1.15
    assert 0.4 <= r.floor_friction <= 1.2
    assert 0.01 <= r.floor_solref <= 0.04
    assert 2.0 <= r.servo.stall_torque <= 2.7
    assert 380 <= math.degrees(r.servo.max_speed) <= 460
    assert 0.7 * nova_model.SERVO_KP <= r.servo.kp <= 1.3 * nova_model.SERVO_KP
    assert 0.7 * nova_model.SERVO_KD <= r.servo.kd <= 1.3 * nova_model.SERVO_KD
    assert 0.005 <= r.servo.dead_time <= 0.040
    assert round(r.servo.dead_time * 1000) % 2 == 0
    assert np.all(np.abs(np.degrees(r.servo.offsets)) <= 3.0)
    assert np.all((0.0 <= np.degrees(r.servo.deadband)) & (np.degrees(r.servo.deadband) <= 2.0))
    assert r.period_jitter == pytest.approx(0.002)


def test_pushes_come_every_3_to_6_seconds():
    r = sample(3, duration=60.0)
    times = [p.time for p in r.pushes]
    assert 3.0 <= times[0] <= 6.0
    assert all(3.0 <= b - a <= 6.0 for a, b in zip(times, times[1:]))
    assert times[-1] < 60.0
    for p in r.pushes:
        assert p.force[2] == 0.0
        assert math.hypot(p.force[0], p.force[1]) <= 15.0
        assert 0.1 <= p.duration <= 0.2


def test_pushes_can_be_switched_off():
    assert sample(3, pushes=False).pushes == []


def test_nominal_values_are_the_model_values():
    r = rnd.Randomization()
    assert r.floor_friction == nova_model.FLOOR_FRICTION
    assert r.servo.stall_torque == nova_model.SERVO_TORQUE
    assert r.pushes == []


def test_apply_changes_the_simulation():
    s = sim.NovaSim()
    torso, leg = s.model.body("torso").id, s.model.body("FL_upper_leg").id
    floor = s.model.geom("floor").id
    mass, leg_mass, com = s.model.body_mass[torso], s.model.body_mass[leg], s.model.body_ipos[torso].copy()
    r = rnd.Randomization(torso_mass=1.1, torso_com=(10.0, -5.0), leg_mass=0.9,
                          floor_friction=0.5, floor_solref=0.03, servo=sim.ServoParams(kp=20.0))
    rnd.apply(s, r)
    assert s.model.body_mass[torso] == pytest.approx(1.1 * mass)
    assert s.model.body_mass[leg] == pytest.approx(0.9 * leg_mass)
    assert s.model.body_ipos[torso] == pytest.approx(com + (0.010, -0.005, 0.0))
    assert s.model.geom_friction[floor, 0] == 0.5
    assert s.model.geom_solref[floor, 0] == 0.03
    assert s.params.kp == 20.0


# ---------------------------------------------------------------------------
# Fehler des Lagesensors
# ---------------------------------------------------------------------------

def standing_sim() -> sim.NovaSim:
    import gait
    from kinematics import body_ik
    s = sim.NovaSim()
    s.reset(body_ik(gait.STAND.feet, gait.STAND.body), gait.STAND.body.height)
    for _ in range(5):
        s.step(body_ik(gait.STAND.feet, gait.STAND.body))
    return s


@pytest.mark.parametrize("seed", range(30))
def test_imu_errors_stay_inside_their_ranges(seed):
    e = sample(seed).imu
    assert 0.0 <= e.gyro_noise <= 0.02
    assert 0.0 <= e.accel_noise <= 0.2
    assert all(abs(b) <= 0.02 for b in e.gyro_bias)
    assert all(abs(m) <= 2.0 for m in e.mount)
    assert 0.0 <= e.delay <= 0.010
    assert round(e.delay * 1000) % 2 == 0


def test_imu_errors_are_the_same_with_and_without_pushes():
    a = rnd.sample(np.random.default_rng(9), pushes=True)
    b = rnd.sample(np.random.default_rng(9), pushes=False)
    assert a.imu == b.imu


def test_nominal_imu_reading_is_the_raw_sensor():
    s = standing_sim()
    assert rnd.imu_reading(s, rnd.ImuErrors(), np.random.default_rng(0)) == s.imu()


def test_gyro_bias_is_added():
    s = standing_sim()
    gyro, _ = rnd.imu_reading(s, rnd.ImuErrors(gyro_bias=(0.01, -0.02, 0.0)), np.random.default_rng(0))
    raw, _ = s.imu()
    assert gyro == pytest.approx((raw[0] + 0.01, raw[1] - 0.02, raw[2]))


def test_tilted_mounting_looks_like_a_tilted_dog():
    # Sensor 2° um die Längsachse verdreht: in Ruhe misst er, als stünde der Hund 2° schief
    import imu
    s = standing_sim()
    _, accel = rnd.imu_reading(s, rnd.ImuErrors(mount=(2.0, 0.0, 0.0)), np.random.default_rng(0))
    f = imu.ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), accel, 0.02)
    assert f.roll - s.torso().roll == pytest.approx(2.0, abs=0.1)


def test_delayed_reading_comes_from_the_past():
    s = standing_sim()
    errors = rnd.ImuErrors(delay=0.006)
    assert rnd.imu_reading(s, errors, np.random.default_rng(0)) == s.imu(delay=0.006)


def test_noise_has_the_given_size():
    s = standing_sim()
    rng = np.random.default_rng(0)
    errors = rnd.ImuErrors(gyro_noise=0.01, accel_noise=0.1)
    raw_gyro, raw_accel = s.imu()
    readings = [rnd.imu_reading(s, errors, rng) for _ in range(2000)]
    gyro = np.array([r[0] for r in readings]) - raw_gyro
    accel = np.array([r[1] for r in readings]) - raw_accel
    assert gyro.std() == pytest.approx(0.01, rel=0.1)
    assert accel.std() == pytest.approx(0.1, rel=0.1)


def test_push_strength_can_be_raised():
    strong = rnd.sample_pushes(np.random.default_rng(0), 600.0, max_force=25.0)
    forces = [math.hypot(p.force[0], p.force[1]) for p in strong]
    assert max(forces) <= 25.0 and max(forces) > 15.0


def test_default_push_strength_stays_15_newton():
    a = rnd.sample_pushes(np.random.default_rng(0), 600.0)
    b = rnd.sample_pushes(np.random.default_rng(0), 600.0, max_force=15.0)
    assert a == b
    assert max(math.hypot(p.force[0], p.force[1]) for p in a) <= 15.0


def test_strong_side_pushes_come_from_the_side():
    style = rnd.PUSH_STYLES["stark-seitlich"]
    pushes = rnd.sample_pushes(np.random.default_rng(0), 600.0, style=style)
    sides = set()
    for p in pushes:
        fx, fy, fz = p.force
        force = math.hypot(fx, fy)
        assert style.force[0] - 1e-9 <= force <= style.force[1] + 1e-9 and fz == 0.0
        assert abs(math.degrees(math.atan2(fx, abs(fy)))) <= style.side_only + 1e-9  # höchstens 30° neben links/rechts
        assert style.duration[0] <= p.duration <= style.duration[1]
        sides.add(fy > 0)
    assert sides == {True, False}  # nach links und nach rechts
    gaps = np.diff([p.time for p in pushes])
    assert gaps.min() >= style.every[0] and gaps.max() <= style.every[1]


def test_push_style_reaches_sample():
    style = rnd.PUSH_STYLES["stark-seitlich"]
    r = rnd.sample(np.random.default_rng(1), True, 60.0, style=style)
    assert r.pushes and min(math.hypot(p.force[0], p.force[1]) for p in r.pushes) >= style.force[0]
