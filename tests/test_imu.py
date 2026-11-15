"""Komplementärfilter: macht aus Drehrate und Beschleunigung die Lage des Rumpfs."""

import math

import pytest

from imu import G, ComplementaryFilter

DT = 0.02


def at_rest(roll: float = 0.0, pitch: float = 0.0):
    """Was der Beschleunigungssensor in Ruhe misst, wenn der Rumpf so geneigt ist (Grad)."""
    r, p = math.radians(roll), math.radians(pitch)
    return (-math.sin(p) * G, math.sin(r) * math.cos(p) * G, math.cos(r) * math.cos(p) * G)


def test_level_and_at_rest_gives_no_tilt():
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    assert f.up == pytest.approx((0.0, 0.0, 1.0))
    assert (f.roll, f.pitch) == pytest.approx((0.0, 0.0))


def test_first_measurement_sets_the_tilt_directly():
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), at_rest(roll=20.0, pitch=-10.0), DT)
    assert (f.roll, f.pitch) == pytest.approx((20.0, -10.0))


def test_nose_down_is_positive_pitch():
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), (-2.0, 0.0, 9.6), DT)  # Schwerkraft zieht nach vorne
    assert f.pitch > 0


def test_gyro_alone_turns_the_estimate():
    # Filter vertraut fast nur dem Gyro: 0,5 rad/s um die Längsachse, 1 s lang
    f = ComplementaryFilter(time_constant=1e9)
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(50):
        f.update((0.5, 0.0, 0.0), at_rest(), DT)
    assert f.roll == pytest.approx(math.degrees(0.5), abs=0.1)
    assert f.pitch == pytest.approx(0.0, abs=0.1)


def test_gyro_pitch_rate_has_the_same_sign_as_pitch():
    f = ComplementaryFilter(time_constant=1e9)
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(50):
        f.update((0.0, 0.3, 0.0), at_rest(), DT)
    assert f.pitch == pytest.approx(math.degrees(0.3), abs=0.1)


def test_accelerometer_stops_the_gyro_from_drifting_away():
    # Gyro zeigt dauernd 0,05 rad/s zu viel. Ohne Korrektur wären das nach 10 s fast 29°.
    f = ComplementaryFilter(time_constant=0.5)
    for _ in range(500):
        f.update((0.05, 0.0, 0.0), at_rest(), DT)
    assert abs(f.roll) < 2.0


def test_estimate_follows_a_real_tilt():
    f = ComplementaryFilter(time_constant=0.5)
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(250):  # 5 s = 10 Zeitkonstanten
        f.update((0.0, 0.0, 0.0), at_rest(roll=15.0), DT)
    assert f.roll == pytest.approx(15.0, abs=0.1)


def test_shocks_are_ignored():
    # Beim Aufsetzen der Füße misst der Sensor kurz ein Vielfaches von g
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(50):
        f.update((0.0, 0.0, 0.0), (25.0, 0.0, 15.0), DT)
    assert (f.roll, f.pitch) == pytest.approx((0.0, 0.0))


def test_up_always_has_length_1():
    f = ComplementaryFilter()
    for i in range(200):
        f.update((0.8, -0.6, 0.4), at_rest(roll=10.0 * math.sin(i / 10)), DT)
    assert math.sqrt(sum(u * u for u in f.up)) == pytest.approx(1.0)


def test_footfall_with_1_2_g_is_ignored():
    # Beim Trab misst der Sensor beim Aufsetzen der Füße bis ca. 1,15 g
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(50):
        f.update((0.0, 0.0, 0.0), tuple(1.2 * a for a in at_rest(roll=20.0)), DT)
    assert f.roll == pytest.approx(0.0)


def test_slightly_off_1_g_is_still_used():
    f = ComplementaryFilter()
    f.update((0.0, 0.0, 0.0), at_rest(), DT)
    for _ in range(250):
        f.update((0.0, 0.0, 0.0), tuple(1.1 * a for a in at_rest(roll=20.0)), DT)
    assert f.roll == pytest.approx(20.0, abs=0.1)
