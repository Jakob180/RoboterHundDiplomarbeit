"""Auswertung des klassischen Trabs."""

import numpy as np
import pytest

import evaluate
import gait
import randomization as rnd
import sim

FRAMES = gait.demo_frames(sim.CONTROL_HZ)


@pytest.fixture(scope="module")
def nominal():
    return evaluate.run_episode(FRAMES, rnd.Randomization())


def test_nominal_trot_walks(nominal):
    assert nominal.fallen_at is None
    assert nominal.distance > 0.8
    assert abs(nominal.drift) < 50.0
    assert nominal.max_tilt < 5.0
    assert 0.0 < nominal.peak_torque < 1.0


def test_imu_filter_follows_the_true_tilt(nominal):
    # gemessen 1,9° (Stöße beim Aufsetzen der Füße)
    assert nominal.imu_error < 3.0


def test_same_randomization_gives_the_same_result():
    r = rnd.sample(np.random.default_rng(4))
    assert evaluate.run_episode(FRAMES, r) == evaluate.run_episode(FRAMES, r)


def test_strong_push_tilts_the_dog_more(nominal):
    pushed = evaluate.run_episode(FRAMES, rnd.Randomization(pushes=[rnd.Push(12.0, (0.0, 10.0, 0.0), 0.2)]))
    assert pushed.max_tilt > nominal.max_tilt + 2.0


def test_falling_is_detected_and_ends_the_episode():
    # Viel zu schwache Servos: Nova sackt sofort zusammen
    result = evaluate.run_episode(FRAMES, rnd.Randomization(servo=sim.ServoParams(stall_torque=0.3)))
    assert result.fallen_at is not None and result.fallen_at < FRAMES[-1].time
    assert result.distance is None


def test_calibration_scenario_uses_the_same_errors_as_plan_section_2_1():
    # damals: np.random.default_rng(1), je Bein uniform(-3, 3, 3)
    old = np.random.default_rng(1)
    expected = np.radians(np.concatenate([old.uniform(-3, 3, 3) for _ in range(4)]))
    assert evaluate.SCENARIOS["Kalibrierfehler ±3°"].servo.offsets == pytest.approx(expected)


def test_summary_counts_falls_and_averages_only_the_walkers():
    ok = evaluate.EpisodeResult(None, 0.9, 10.0, -2.0, 3.0, 0.6, 1.0)
    fell = evaluate.EpisodeResult(8.0, None, None, None, 40.0, 0.9, 5.0)
    summary = evaluate.summarize([ok, fell, ok._replace(distance=0.7, drift=-30.0, yaw=4.0)])
    assert summary["Episoden"] == 3
    assert summary["umgefallen"] == 1
    assert summary["Strecke Mittel"] == pytest.approx(0.8)
    assert summary["Strecke min"] == pytest.approx(0.7)
    assert summary["|Drift| Mittel [mm]"] == pytest.approx(20.0)
    assert summary["|Gier| max [°]"] == pytest.approx(4.0)
    assert summary["Moment max"] == pytest.approx(0.9)


def test_phase1_baseline_stays_reproducible():
    # Durchlauf 0 der Gruppe "ohne Stöße" aus model/results/trab-phase1.json
    import json
    from pathlib import Path
    saved = json.loads((Path(evaluate.__file__).parent / "results" / "trab-phase1.json").read_text(encoding="utf-8"))
    first = saved["Randomisiert"]["ohne Stöße"]["Episoden"][0]
    r = rnd.sample(np.random.default_rng((saved["Einstellungen"]["seed"], 0)), False, FRAMES[-1].time)
    again = evaluate.run_episode(FRAMES, r)._asdict()
    for key, value in first.items():
        assert again[key] == pytest.approx(value, abs=1e-3), key
