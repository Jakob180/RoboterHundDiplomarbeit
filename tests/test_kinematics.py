"""Inverse Kinematik: hin und zurück, ganzer Körper und einzelnes Bein."""

import numpy as np
import pytest

import gait
from kinematics import LEGS, BodyPose, body_ik, body_points, foot_ik, leg_fk, leg_ik


@pytest.mark.parametrize("seed", range(5))
def test_leg_ik_and_fk_go_there_and_back(seed):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        point = (rng.uniform(-80, 80), rng.uniform(40, 80), rng.uniform(-210, -120))
        assert leg_fk(leg_ik(*point)) == pytest.approx(point, abs=1e-6)


@pytest.mark.parametrize("body", [gait.STAND.body, BodyPose(150.0, roll=5.0, pitch=-8.0, yaw=3.0)])
def test_body_ik_puts_the_feet_where_they_were_asked(body):
    feet = gait.STAND.feet
    points = body_points(body_ik(feet, body), body)
    for leg in LEGS:
        assert points[leg][-1] == pytest.approx(feet[leg], abs=1e-6)


def test_foot_ik_is_body_ik_for_one_leg():
    body = BodyPose(160.0, roll=3.0, pitch=-5.0)
    full = body_ik(gait.STAND.feet, body)
    for leg in LEGS:
        assert foot_ik(leg, gait.STAND.feet[leg], body) == pytest.approx(full[leg])


def test_unreachable_foot_names_the_leg():
    with pytest.raises(ValueError, match="RL"):
        foot_ik("RL", (-93.0, 94.0, -400.0), gait.STAND.body)
