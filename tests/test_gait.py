"""Trab: geradeaus unverändert, dazu Drehen und die öffentliche Phase."""

import math

import pytest

import gait
import servos
from kinematics import LEGS, body_ik

DT = 0.02

# Fußpunkte des Trabs VOR dem Einbau der Drehrate: 40 Schritte mit 150 mm/s,
# dann 80 mm/s. Geradeaus darf sich daran nichts ändern.
GOLDEN = {
    0: ({"FL": (90.0, 94.0, 0.0), "FR": (90.0, -94.0, 0.0),
         "RL": (-96.0, 94.0, 0.0), "RR": (-96.0, -94.0, 0.0)}, 3.0),
    13: ({"FL": (51.0, 94.0, 0.0), "FR": (111.0, -94.0, 0.0),
          "RL": (-75.0, 94.0, 0.0), "RR": (-135.0, -94.0, 0.0)}, 42.0),
    27: ({"FL": (106.5, 94.0, 0.0), "FR": (69.061381, -94.0, 0.611794),
          "RL": (-116.938619, 94.0, 0.611794), "RR": (-79.5, -94.0, 0.0)}, 84.0),
    41: ({"FL": (69.75605, 94.0, 8.637288), "FR": (104.8, -94.0, 0.0),
          "RL": (-81.2, 94.0, 0.0), "RR": (-116.24395, -94.0, 8.637288)}, 123.2),
    59: ({"FL": (89.0, 94.0, 0.0), "FR": (106.002113, -94.0, 12.5),
          "RL": (-79.997887, 94.0, 12.5), "RR": (-97.0, -94.0, 0.0)}, 152.0),
}


def world(trot: gait.Trot, foot) -> tuple[float, float]:
    """Fußpunkt aus dem Koordinatensystem des Hundes in die Welt umrechnen."""
    h = math.radians(trot.heading)
    return (trot.x + foot[0] * math.cos(h) - foot[1] * math.sin(h),
            trot.y + foot[0] * math.sin(h) + foot[1] * math.cos(h))


def test_straight_trot_gives_the_same_foot_points_as_before():
    trot = gait.Trot()
    for i in range(60):
        pose = trot.update(DT, 150.0 if i < 40 else 80.0)
        if i in GOLDEN:
            feet, distance = GOLDEN[i]
            assert trot.distance == pytest.approx(distance)
            for leg in LEGS:
                assert pose.feet[leg] == pytest.approx(feet[leg], abs=1e-5), (i, leg)


def test_phase_is_public_and_runs_from_0_to_1_per_period():
    trot = gait.Trot(period=0.5)
    assert trot.phase == 0.0
    trot.update(DT, 0.0)
    assert trot.phase == pytest.approx(0.04)
    for _ in range(24):
        trot.update(DT, 0.0)
    assert trot.phase == pytest.approx(0.0, abs=1e-9) or trot.phase == pytest.approx(1.0, abs=1e-9)


def test_turning_on_the_spot_changes_only_the_heading():
    trot = gait.Trot()
    for _ in range(50):
        trot.update(DT, 0.0, yaw_rate=30.0)
    assert trot.heading == pytest.approx(30.0)
    assert (trot.x, trot.y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_positive_yaw_rate_walks_a_left_curve():
    # 100 mm/s und 90 °/s: nach 1 s ein Viertelkreis mit Radius v/ω = 63,7 mm
    trot = gait.Trot()
    for _ in range(50):
        trot.update(DT, 100.0, yaw_rate=90.0)
    radius = 100.0 / math.radians(90.0)
    assert trot.heading == pytest.approx(90.0)
    assert (trot.x, trot.y) == pytest.approx((radius, radius), abs=1e-6)
    assert trot.distance == pytest.approx(100.0)


def test_feet_on_the_ground_do_not_slide_while_turning():
    trot = gait.Trot()
    before = None
    for i in range(150):
        pose = trot.update(DT, 120.0, yaw_rate=25.0 if i < 100 else -25.0)
        now = {leg: (world(trot, pose.feet[leg]), pose.feet[leg][2]) for leg in LEGS}
        if before is not None:
            for leg in LEGS:
                if now[leg][1] == 0.0 and before[leg][1] == 0.0:  # stand zweimal am Boden
                    assert now[leg][0] == pytest.approx(before[leg][0], abs=1e-6), (i, leg)
        before = now


def test_feet_come_back_to_the_base_pose_after_turning_stops():
    trot = gait.Trot()
    for _ in range(50):
        trot.update(DT, 100.0, yaw_rate=30.0)
    for _ in range(50):
        pose = trot.update(DT, 0.0, yaw_rate=0.0)
    for leg in LEGS:
        assert pose.feet[leg] == pytest.approx(gait.STAND.feet[leg], abs=1e-6)


@pytest.mark.parametrize("speed, yaw_rate", [(200.0, 30.0), (200.0, -30.0), (0.0, 30.0), (100.0, -30.0)])
def test_turning_trot_stays_inside_the_joint_limits(speed, yaw_rate):
    # Befehlsbereich laut Plan: 0..200 mm/s und ±0,5 rad/s (rund ±30 °/s)
    trot = gait.Trot()
    for _ in range(150):
        pose = trot.update(DT, speed, yaw_rate=yaw_rate)
        angles = body_ik(pose.feet, pose.body)
        for name in servos.JOINT_ORDER:
            low, high = servos.kinematic_limits(name)
            assert low <= getattr(angles[name[:2]], name[3:]) <= high, name


def test_swing_tells_how_far_each_foot_is_in_its_swing():
    trot = gait.Trot()
    swings = {leg: [] for leg in ("FL", "FR", "RL", "RR")}
    for _ in range(250):  # 10 Perioden
        trot.update(0.02, 150.0)
        for leg in swings:
            swings[leg].append(trot.swing(leg))
    for leg, values in swings.items():
        in_air = [u for u in values if u is not None]
        assert len(in_air) / len(values) == pytest.approx(1 - trot.duty, abs=0.02)
        assert all(0.0 <= u < 1.0 for u in in_air)
    assert [u is None for u in swings["FL"]] == [u is None for u in swings["RR"]]  # diagonale Paare
    assert [u is None for u in swings["FR"]] == [u is None for u in swings["RL"]]
    assert swings["FL"] != swings["FR"]


def test_lying_pose_is_low_and_inside_the_joint_limits():
    angles = body_ik(gait.LIE.feet, gait.LIE.body)
    for leg in LEGS:
        for joint in ("shoulder", "upper", "lower"):
            low, high = servos.kinematic_limits(f"{leg}_{joint}")
            assert low <= getattr(angles[leg], joint) <= high
    assert gait.LIE.body.height < gait.SIT.body.height < gait.STAND.body.height
