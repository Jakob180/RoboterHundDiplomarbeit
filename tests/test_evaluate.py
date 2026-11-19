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


# ---------------------------------------------------------------------------
# Vergleich Trab ohne Netz gegen Trab mit Netz (Phase 3)
# ---------------------------------------------------------------------------

def test_trot_without_network_walks_in_the_comparison():
    result = evaluate.run_policy_episode(rnd.Randomization())
    assert result.fallen_at is None
    assert result.distance > 0.7
    assert result.max_tilt < 10.0


def test_network_that_outputs_zero_is_the_trot():
    r = evaluate.POLICY_SCENARIOS["Kalibrierfehler ±3°"]
    assert evaluate.run_policy_episode(r, lambda obs: np.zeros(12)) == evaluate.run_policy_episode(r)


def test_strong_push_in_the_comparison_tilts_more():
    calm = evaluate.run_policy_episode(rnd.Randomization())
    pushed = evaluate.run_policy_episode(evaluate.POLICY_SCENARIOS["Seitenstoß 15 N"])
    assert pushed.max_tilt > calm.max_tilt + 2.0


def test_compare_runs_trot_and_network_on_the_same_dogs():
    out = evaluate.compare(lambda obs: np.zeros(12), episodes=1, scenarios={"Standard": rnd.Randomization()})
    assert out["Szenarien"]["Standard"]["Trab"] == out["Szenarien"]["Standard"]["Netz"]
    assert out["Randomisiert"]["Trab"] == out["Randomisiert"]["Netz"]
    assert out["Randomisiert"]["Trab"]["Episoden"] == 1


def test_acceptance_criteria_from_the_plan():
    def result(fallen=None, yaw=0.0):
        return evaluate.PolicyResult(fallen, None if fallen else 0.9, 0.0, None if fallen else yaw, 5.0)
    out = {
        "Szenarien": {
            "Seitenstoß 20 N": {"Trab": result(fallen=6.0)._asdict(), "Netz": result()._asdict()},
            "Kalibrierfehler ±3°": {"Trab": result(yaw=40.0)._asdict(), "Netz": result(yaw=-15.0)._asdict()},
        },
        "Randomisiert": {"Trab": {"umgefallen": 6}, "Netz": {"umgefallen": 2}},
    }
    assert evaluate.acceptance(out) == {evaluate.ACCEPT_FALLS: True, evaluate.ACCEPT_PUSH: True,
                                        evaluate.ACCEPT_CALIBRATION: True}
    out["Randomisiert"]["Netz"]["umgefallen"] = 4    # nur ein Drittel weniger: nicht "deutlich"
    out["Szenarien"]["Kalibrierfehler ±3°"]["Netz"] = result(yaw=25.0)._asdict()  # nicht halbiert
    out["Szenarien"]["Seitenstoß 20 N"]["Netz"] = result(fallen=6.2)._asdict()
    assert evaluate.acceptance(out) == {evaluate.ACCEPT_FALLS: False, evaluate.ACCEPT_PUSH: False,
                                        evaluate.ACCEPT_CALIBRATION: False}


def test_networks_load_from_zip_and_onnx(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnxruntime")
    import json
    from stable_baselines3 import PPO
    import export_onnx
    import policy
    from nova_env import NovaEnv
    env = NovaEnv(randomize=False, pushes=False)
    PPO("MlpPolicy", env, device="cpu", seed=0, policy_kwargs=dict(net_arch=[8])).save(tmp_path / "final.zip")
    (tmp_path / "settings.json").write_text(json.dumps(policy.settings(env.policy)), encoding="utf-8")
    export_onnx.export(tmp_path / "final.zip", tmp_path / "netz.onnx")
    obs = np.random.default_rng(0).normal(size=34).astype(np.float32)
    from_zip, mode_zip = evaluate.load_network(tmp_path / "final.zip")
    from_onnx, mode_onnx = evaluate.load_network(tmp_path / "netz.onnx")
    assert mode_zip == mode_onnx == "residual"
    assert from_zip(obs) == pytest.approx(np.clip(from_onnx(obs), -1, 1), abs=1e-5)
