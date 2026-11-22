"""Kalibrier-Programm: jedes Gelenk einzeln auf eine Bezugslage fahren, Richtung und Grenzen festlegen."""

import json

import pytest

import calibrate
import gait
import servos
from kinematics import body_ik


class FakeServos:
    """Merkt sich die rohen Winkel je Kanal (statt den PCA9685 anzusteuern)."""

    def __init__(self) -> None:
        self.raw: dict[int, float | None] = {}
        self.history: list[tuple[int, float]] = []

    def set_raw(self, channel: int, angle: float) -> float:
        angle = min(servos.RAW_RANGE[1], max(servos.RAW_RANGE[0], angle))
        self.raw[channel] = angle
        self.history.append((channel, angle))
        return angle

    def release(self, channel: int) -> None:
        self.raw[channel] = None


def session(commands: list[str]):
    answers = iter(commands)
    said: list[str] = []
    hw = FakeServos()
    s = calibrate.Session(hw, ask=lambda prompt="": next(answers), say=said.append, sleep=lambda t: None)
    return s, hw, said


def angles(hw: FakeServos, channel: int) -> list[float]:
    return [a for ch, a in hw.history if ch == channel]


def test_left_upper_leg_mounted_normally():
    # Horn in der Stehpose aufgesteckt (roh 90°). Oberschenkel senkrecht bei roh 140°.
    # +5° roh schwingt ihn nach vorne (= Kinematik +). Anschläge bei roh 35° und 165°.
    commands = ["+10"] * 5 + ["ref", "dir", "j"] + ["-10"] * 10 + ["-5", "min"] + ["+10"] * 13 + ["max", "ok"]
    s, hw, said = session(commands)
    joint = s.calibrate_joint("FL_upper")
    assert hw.history[0] == (1, 90.0)                       # Start bei 90°, so wurde das Horn aufgesteckt
    assert joint.channel == 1 and joint.inverted is False
    assert joint.zero == pytest.approx(140.0)               # Oberschenkel senkrecht = Kinematik 0
    assert joint.min_angle == pytest.approx(35.0 + calibrate.MARGIN)
    assert joint.max_angle == pytest.approx(165.0 - calibrate.MARGIN)
    stand = body_ik(gait.STAND.feet, gait.STAND.body)["FL"].upper   # Stehpose: Oberschenkel ca. 48° nach hinten
    assert joint.neutral == pytest.approx(140.0 + stand, abs=0.1)
    assert hw.raw[1] is None                                # danach locker
    assert not any("außerhalb" in text for text in said)


def test_right_lower_leg_mounted_mirrored_is_inverted():
    # +5° roh bewegt das Knie in die falsche Richtung -> gespiegelt eingebaut
    s, _, _ = session(["ref", "dir", "n"] + ["-10"] * 3 + ["min"] + ["+10"] * 6 + ["max", "ok"])
    joint = s.calibrate_joint("FR_lower")
    assert joint.inverted is True
    # Bezug Knie 90° bei roh 90°: logisch 180 - 90 = 90 = Nullpunkt + 90 -> Nullpunkt 0
    assert joint.zero == pytest.approx(0.0)
    assert (joint.min_angle, joint.max_angle) == pytest.approx((60.0 + calibrate.MARGIN, 120.0 - calibrate.MARGIN))
    assert joint.neutral == pytest.approx(89.0, abs=1.0)


def test_direction_check_moves_and_comes_back():
    s, hw, said = session(["dir", "j", "ref", "-10", "min", "+10", "+10", "max", "ok"])
    s.calibrate_joint("RL_shoulder")
    moved = angles(hw, 6)
    assert max(moved[:12]) == pytest.approx(90.0 + calibrate.DIRECTION_TEST)
    assert moved[2 * int(calibrate.DIRECTION_TEST)] == pytest.approx(90.0)  # wieder zurück
    assert any("außen" in text for text in said)           # erklärt, wohin + bei der Schulter geht


def test_moves_are_gentle():
    s, hw, _ = session(["+10", "-10", "ref", "dir", "j", "-10", "min", "+10", "+10", "max", "ok"])
    s.calibrate_joint("FL_shoulder")
    moved = angles(hw, 0)
    assert max(abs(b - a) for a, b in zip(moved, moved[1:])) <= 1.0 + 1e-9


def test_big_jumps_are_refused():
    s, hw, said = session(["+45", "ref", "dir", "j", "-10", "min", "+10", "+10", "max", "ok"])
    s.calibrate_joint("FL_shoulder")
    assert any("höchstens" in text for text in said)
    assert max(angles(hw, 0)) <= 100.0


def test_cannot_finish_before_everything_is_measured():
    s, _, said = session(["ok", "ref", "dir", "j", "-10", "min", "+10", "+10", "max", "ok"])
    assert s.calibrate_joint("FL_shoulder") is not None
    assert any("fehlt" in text for text in said)


def test_limits_too_close_together_are_refused():
    s, _, said = session(["ref", "dir", "j", "min", "+5", "max", "ok", "-10", "min", "+10", "+10", "max", "ok"])
    joint = s.calibrate_joint("FL_shoulder")
    assert any("eng" in text for text in said)
    assert joint.max_angle - joint.min_angle > 10.0


def test_warns_when_the_standing_pose_is_outside_the_limits():
    # Oberschenkel senkrecht schon bei roh 90° -> Stehpose bräuchte roh 40°, Anschlag aber bei 70°
    s, _, said = session(["ref", "dir", "j", "-10", "-10", "min"] + ["+10"] * 4 + ["max", "ok"])
    joint = s.calibrate_joint("RR_upper")
    assert any("Stehpose" in text and "außerhalb" in text for text in said)
    assert joint.min_angle <= joint.neutral <= joint.max_angle


def test_quit_returns_nothing():
    s, hw, _ = session(["+3", "q"])
    assert s.calibrate_joint("FL_shoulder") is None
    assert hw.raw[0] is None


def test_results_are_saved_and_read_by_servos(tmp_path):
    s, _, _ = session(["ref", "dir", "j", "-10", "-10", "min"] + ["+10"] * 4 + ["max", "ok"])
    joint = s.calibrate_joint("RR_upper")
    path = tmp_path / "kalibrierung.json"
    calibrate.save(path, {"RR_upper": joint})
    assert servos.load_calibration(path, servos.JOINTS)["RR_upper"] == joint
    assert "datum" in json.loads(path.read_text(encoding="utf-8"))
    calibrate.save(path, {"FL_lower": servos.JOINTS["FL_lower"]})  # zweiter Durchgang ergänzt nur
    assert set(json.loads(path.read_text(encoding="utf-8"))["joints"]) == {"RR_upper", "FL_lower"}


def test_real_servos_only_with_hardware_flag(monkeypatch, capsys):
    monkeypatch.setattr(calibrate, "SIM_MODE", False)
    assert calibrate.main(["--joint", "FL_upper"]) == 2
    assert "--hardware" in capsys.readouterr().out


def test_unknown_joint_is_refused(capsys):
    assert calibrate.main(["--joint", "XX_knie"]) == 2
