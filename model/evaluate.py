"""
Kennzahlen des klassischen Trabs: der Vergleichswert für das neuronale Netz.

Spielt den Ablauf aus gait.demo_frames() (Stehen, Sitzen, Trab 150 mm/s,
Anhalten) in der Simulation ab, und zwar
1. in festen Szenarien wie im Plan Abschnitt 2.1 (Reibung, Kalibrierfehler, Stöße ...),
2. in vielen zufällig veränderten Durchläufen (Domain Randomization), einmal
   ohne und einmal mit Stößen. Feste Zufallszahlen: gleicher --seed, gleiche Zahlen.

    python model/evaluate.py                       # 20 Durchläufe je Gruppe
    python model/evaluate.py --episodes 50 --json model/results/trab.json

Der offene Trab reagiert nicht auf die IMU. Der IMU-Fehler zeigt nur, wie gut
der Filter aus src/imu.py die echte Lage trifft (wichtig fürs Netz später).

Phase 3: trainiertes Netz gegen den Trab ohne Netz, mit denselben Hunden,
Stößen und Zufallszahlen, 10-s-Episoden mit 150 mm/s geradeaus:

    python model/evaluate.py --policy checkpoints/<lauf>/final.zip
    python model/evaluate.py --policy model/policies/nova_residual_v1.onnx

Am Ende stehen die Abnahme-Kriterien aus dem Plan (Phase 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gait import Frame, demo_frames  # noqa: E402
from imu import ComplementaryFilter  # noqa: E402
import config  # noqa: E402
from kinematics import LEGS  # noqa: E402
from nova_env import NovaEnv  # noqa: E402
from policy import N_ACTIONS, OnnxNetwork, check_settings  # noqa: E402
from randomization import Push, Randomization, apply, sample  # noqa: E402
from sim import CONTROL_HZ, NovaSim, ServoParams  # noqa: E402


class EpisodeResult(NamedTuple):
    fallen_at: float | None  # s, wann Nova umgefallen ist (None = nicht umgefallen)
    distance: float | None   # gelaufene Strecke als Anteil der geplanten (1.0 = 100 %)
    drift: float | None      # mm seitlich abgedriftet (+ = nach links)
    yaw: float | None        # Grad gedreht (+ = nach links)
    max_tilt: float          # Grad, größte Abweichung von geplantem Roll bzw. Pitch
    peak_torque: float       # größtes Servo-Moment als Anteil des Blockiermoments
    imu_error: float         # Grad, größter Fehler des IMU-Filters gegenüber der echten Lage


# s: so lange zählt der IMU-Fehler noch nicht. Direkt nach reset() setzt sich der Hund
# noch auf den Boden, und der Filter übernimmt den ersten (verwackelten) Messwert. Am
# echten Hund wird der Filter im Stillstand gestartet.
IMU_WARMUP = 1.0


def run_episode(frames: list[Frame], r: Randomization) -> EpisodeResult:
    """Einen Durchlauf rechnen. Endet sofort, wenn Nova umfällt."""
    sim = NovaSim()
    apply(sim, r)
    sim.reset(frames[0].angles, frames[0].pose.body.height)
    jitter = np.random.default_rng(r.seed)
    imu = ComplementaryFilter()
    pushes = sorted(r.pushes)
    tilt = torque = imu_error = 0.0

    for frame in frames:
        while pushes and pushes[0].time <= sim.data.time + 1e-9:
            push = pushes.pop(0)
            sim.push(push.force, push.duration)
        before = sim.data.time
        sim.step(frame.angles, 1.0 / CONTROL_HZ + jitter.uniform(-r.period_jitter, r.period_jitter))

        state = sim.torso()
        plan = frame.pose.body
        gyro, accel = sim.imu()
        imu.update(gyro, accel, sim.data.time - before)
        tilt = max(tilt, abs(state.roll - plan.roll), abs(state.pitch - plan.pitch))
        torque = max(torque, float(np.max(np.abs(sim.data.actuator_force) / r.servo.stall_torque)))
        if sim.data.time >= IMU_WARMUP:
            imu_error = max(imu_error, abs(imu.roll - state.roll), abs(imu.pitch - state.pitch))
        if state.z < 60 or abs(state.roll) > 45 or abs(state.pitch - plan.pitch) > 45:
            return EpisodeResult(round(sim.data.time, 3), None, None, None, tilt, torque, imu_error)

    end = sim.torso()
    return EpisodeResult(None, end.x / frames[-1].distance, end.y, end.yaw, tilt, torque, imu_error)


def summarize(results: list[EpisodeResult]) -> dict[str, float | int | None]:
    """Zusammenfassung vieler Durchläufe. Strecke, Drift, Neigung und IMU nur von denen, die stehen blieben."""
    walkers = [r for r in results if r.fallen_at is None]

    def mean(values):
        return float(np.mean(values)) if values else None

    def largest(values):
        return float(np.max(values)) if values else None

    return {
        "Episoden": len(results),
        "umgefallen": len(results) - len(walkers),
        "Strecke Mittel": mean([w.distance for w in walkers]),
        "Strecke min": float(min(w.distance for w in walkers)) if walkers else None,
        "|Drift| Mittel [mm]": mean([abs(w.drift) for w in walkers]),
        "|Drift| max [mm]": largest([abs(w.drift) for w in walkers]),
        "|Gier| Mittel [°]": mean([abs(w.yaw) for w in walkers]),
        "|Gier| max [°]": largest([abs(w.yaw) for w in walkers]),
        "Neigung Mittel [°]": mean([w.max_tilt for w in walkers]),
        "Neigung max [°]": largest([w.max_tilt for w in walkers]),
        "Moment max": largest([r.peak_torque for r in results]),
        "IMU-Fehler max [°]": largest([w.imu_error for w in walkers]),
    }


# ---------------------------------------------------------------------------
# Feste Szenarien (wie Plan Abschnitt 2.1)
# ---------------------------------------------------------------------------

def _calibration_errors() -> np.ndarray:
    """Dieselben zufälligen ±3° wie bei der Messung im Plan (Zufallszahl 1, je Bein 3 Werte)."""
    rng = np.random.default_rng(1)
    return np.radians(np.concatenate([rng.uniform(-3.0, 3.0, 3) for _ in LEGS]))


PUSH_TIME = 12.0  # s, mitten im Trab mit voller Geschwindigkeit


def _side_push(force: float) -> list[Push]:
    return [Push(PUSH_TIME, (0.0, force, 0.0), 0.2)]


SCENARIOS: dict[str, Randomization] = {
    "Standard": Randomization(),
    "Totzeit 40 ms": Randomization(servo=ServoParams(dead_time=0.040)),
    "Reibung 0,4": Randomization(floor_friction=0.4),
    "Reibung 0,3": Randomization(floor_friction=0.3),
    "Reibung 0,15": Randomization(floor_friction=0.15),
    "Kalibrierfehler ±3°": Randomization(servo=ServoParams(offsets=_calibration_errors())),
    "Spiel 2° an allen Gelenken": Randomization(servo=ServoParams(deadband=np.radians(2.0))),
    "Seitenstoß 5 N / 0,2 s": Randomization(pushes=_side_push(5.0)),
    "Seitenstoß 10 N / 0,2 s": Randomization(pushes=_side_push(10.0)),
    "Seitenstoß 15 N / 0,2 s": Randomization(pushes=_side_push(15.0)),
    "Seitenstoß 20 N / 0,2 s": Randomization(pushes=_side_push(20.0)),
    "Servo 2,0 N·m (Akku leer)": Randomization(servo=ServoParams(stall_torque=2.0)),
    "Rumpf +20 % (2,9 kg)": Randomization(torso_mass=1.2),
}


# ---------------------------------------------------------------------------
# Phase 3: Trab ohne Netz gegen Trab mit Netz
# ---------------------------------------------------------------------------

COMMAND = (150.0, 0.0)  # mm/s und °/s, wie im Ablauf aus Phase 1
ENV_PUSH_TIME = 5.5     # s ab Start der Simulation (nach 0,5 s Stehen), mitten im Lauf

ACCEPT_FALLS = "Randomisiert mit Stößen: höchstens halb so oft umgefallen"
ACCEPT_PUSH = "Seitenstoß 20 N überstanden"
ACCEPT_CALIBRATION = "Kalibrierfehler ±3°: höchstens halb so stark abgedreht"

Network = Callable[[np.ndarray], np.ndarray]


class PolicyResult(NamedTuple):
    fallen_at: float | None  # s, wann Nova umgefallen ist (None = nicht umgefallen)
    distance: float | None   # gelaufene Strecke als Anteil des Befehls
    drift: float | None      # mm seitlich abgedriftet (+ = nach links)
    yaw: float | None        # Grad gedreht (+ = nach links)
    max_tilt: float          # Grad, größte Neigung (Roll oder Pitch)


def _env_push(force: float) -> list[Push]:
    return [Push(ENV_PUSH_TIME, (0.0, force, 0.0), 0.2)]


POLICY_SCENARIOS: dict[str, Randomization] = {
    "Standard": Randomization(),
    "Kalibrierfehler ±3°": SCENARIOS["Kalibrierfehler ±3°"],
    "Spiel 2°": Randomization(servo=ServoParams(deadband=np.radians(2.0))),
    "Reibung 0,4": Randomization(floor_friction=0.4),
    "Seitenstoß 15 N": Randomization(pushes=_env_push(15.0)),
    "Seitenstoß 20 N": Randomization(pushes=_env_push(20.0)),
    "Seitenstoß 25 N": Randomization(pushes=_env_push(25.0)),
}


def run_policy_episode(
    r: Randomization, network: Network | None = None, mode: str = "residual", command=COMMAND,
) -> PolicyResult:
    """Eine Episode in der Trainingsumgebung. network=None: Trab ohne Netz."""
    env = NovaEnv(mode=mode if network is not None else "residual")
    obs, _ = env.reset(seed=0, options={"randomization": r, "command": command})
    tilt = 0.0
    while True:
        action = np.zeros(N_ACTIONS) if network is None else network(obs)
        obs, _, fallen, truncated, _ = env.step(action)
        state = env.sim.torso()
        tilt = max(tilt, abs(state.roll), abs(state.pitch))
        if fallen:
            return PolicyResult(round(env.sim.data.time, 3), None, None, None, tilt)
        if truncated:
            return PolicyResult(None, env.walked / env.commanded, state.y, state.yaw, tilt)


def _summary(results: list[PolicyResult]) -> dict[str, float | int | None]:
    walkers = [r for r in results if r.fallen_at is None]

    def mean(values):
        return float(np.mean(values)) if values else None

    return {
        "Episoden": len(results),
        "umgefallen": len(results) - len(walkers),
        "Strecke Mittel": mean([w.distance for w in walkers]),
        "|Drift| Mittel [mm]": mean([abs(w.drift) for w in walkers]),
        "|Gier| Mittel [°]": mean([abs(w.yaw) for w in walkers]),
        "Neigung max Mittel [°]": mean([w.max_tilt for w in walkers]),
    }


def compare(
    network: Network, mode: str = "residual", episodes: int = 20, seed: int = 0,
    scenarios: dict[str, Randomization] = POLICY_SCENARIOS,
) -> dict:
    """Trab und Netz in denselben Szenarien und auf denselben zufälligen Hunden (mit Stößen)."""
    out: dict = {"Szenarien": {}, "Randomisiert": {}}
    for name, r in scenarios.items():
        out["Szenarien"][name] = {
            "Trab": run_policy_episode(r)._asdict(),
            "Netz": run_policy_episode(r, network, mode)._asdict(),
        }
    duration = config.EPISODE_SECONDS + config.WARMUP_SECONDS
    dogs = [sample(np.random.default_rng((seed, i)), True, duration) for i in range(episodes)]
    out["Randomisiert"]["Trab"] = _summary([run_policy_episode(r) for r in dogs])
    out["Randomisiert"]["Netz"] = _summary([run_policy_episode(r, network, mode) for r in dogs])
    return out


def acceptance(out: dict) -> dict[str, bool]:
    """Abnahme-Kriterien aus dem Plan (Phase 3), mit 20 N statt 15 N (15 N schafft schon der Trab)."""
    trab, netz = out["Randomisiert"]["Trab"]["umgefallen"], out["Randomisiert"]["Netz"]["umgefallen"]
    falls = netz <= trab / 2 if trab > 0 else netz == 0
    push = out["Szenarien"].get("Seitenstoß 20 N")
    pushed = push is not None and push["Netz"]["fallen_at"] is None
    cal = out["Szenarien"].get("Kalibrierfehler ±3°")
    if cal is None or cal["Netz"]["fallen_at"] is not None:
        turned = False
    elif cal["Trab"]["yaw"] is None:  # Trab umgefallen, Netz nicht
        turned = True
    else:
        turned = abs(cal["Netz"]["yaw"]) <= abs(cal["Trab"]["yaw"]) / 2
    return {ACCEPT_FALLS: falls, ACCEPT_PUSH: pushed, ACCEPT_CALIBRATION: turned}


def load_network(path) -> tuple[Network, str]:
    """Netz aus .zip (Training, braucht stable-baselines3) oder .onnx laden. Gibt (Netz, Modus) zurück."""
    path = Path(path)
    if path.suffix == ".onnx":
        net = OnnxNetwork(path)
        return net, net.mode
    if path.suffix == ".zip":
        from stable_baselines3 import PPO

        meta = json.loads((path.parent / "settings.json").read_text(encoding="utf-8"))
        check_settings(meta, path.name)
        model = PPO.load(path, device="cpu")
        return (lambda obs: model.predict(obs, deterministic=True)[0]), meta["mode"]
    raise ValueError(f"Unbekanntes Format: {path.name} (erwartet .zip oder .onnx)")


def _cell(r: dict) -> str:
    if r["fallen_at"] is not None:
        return f"fällt bei {r['fallen_at']:4.1f} s".ljust(31)
    return f"{100 * r['distance']:4.0f} %  {r['drift']:+5.0f} mm / {r['yaw']:+5.1f}°  {r['max_tilt']:4.1f}°"


def print_comparison(out: dict) -> None:
    print(f"  {'Szenario':22s} {'Trab ohne Netz':31s} | Trab mit Netz")
    print(f"  {'':22s} {'Strecke, seitlich/gedreht, Neig.':31s} |")
    for name, pair in out["Szenarien"].items():
        print(f"  {name:22s} {_cell(pair['Trab'])} | {_cell(pair['Netz'])}")
    n = out["Randomisiert"]["Trab"]["Episoden"]
    print(f"\nRandomisiert mit Stößen ({n} gleiche Hunde):")
    for key in out["Randomisiert"]["Trab"]:
        a, b = out["Randomisiert"]["Trab"][key], out["Randomisiert"]["Netz"][key]
        show = (lambda v: "–" if v is None else f"{v:.3g}")
        print(f"  {key:24s} Trab {show(a):>7s} | Netz {show(b):>7s}")
    print("\nAbnahme (Plan Phase 3):")
    for text, ok in acceptance(out).items():
        print(f"  [{'x' if ok else ' '}] {text}")


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def _row(name: str, r: EpisodeResult) -> str:
    if r.fallen_at is not None:
        walked = f"JA, {r.fallen_at:4.1f} s".ljust(29)
    else:
        walked = f"nein  {100 * r.distance:4.0f} %  {r.drift:+5.0f} mm / {r.yaw:+5.1f}°"
    return (f"  {name:28s} {walked}  {r.max_tilt:5.1f}°  {100 * r.peak_torque:4.0f} %  {r.imu_error:4.1f}°")


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Kennzahlen des klassischen Trabs")
    parser.add_argument("--episodes", type=int, default=20, help="zufällige Durchläufe je Gruppe")
    parser.add_argument("--seed", type=int, default=0, help="Startwert der Zufallszahlen")
    parser.add_argument("--json", help="Ergebnisse zusätzlich als JSON-Datei speichern")
    parser.add_argument("--policy", help="Netz (.zip oder .onnx) gegen den Trab vergleichen")
    args = parser.parse_args()

    if args.policy:
        network, mode = load_network(args.policy)
        print(f"Vergleich: {args.policy} (Modus {mode}), 10 s mit {COMMAND[0]:.0f} mm/s geradeaus\n")
        out = compare(network, mode, args.episodes, args.seed)
        print_comparison(out)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(_rounded(out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"\nGespeichert: {args.json}")
        return

    frames = demo_frames(CONTROL_HZ)
    out: dict = {"Einstellungen": {"episodes": args.episodes, "seed": args.seed}, "Szenarien": {}, "Randomisiert": {}}

    print(f"Klassischer Trab, geplant {frames[-1].distance:.0f} mm")
    print(f"  {'Versuch':28s} {'umgefallen / Strecke / seitlich / gedreht':29s}  Neig.  Moment  IMU")
    for name, r in SCENARIOS.items():
        result = run_episode(frames, r)
        out["Szenarien"][name] = result._asdict()
        print(_row(name, result))

    for group, pushes in (("ohne Stöße", False), ("mit Stößen", True)):
        results = [
            run_episode(frames, sample(np.random.default_rng((args.seed, i)), pushes, frames[-1].time))
            for i in range(args.episodes)
        ]
        summary = summarize(results)
        out["Randomisiert"][group] = {"Zusammenfassung": summary, "Episoden": [r._asdict() for r in results]}
        print(f"\nRandomisiert {group} ({args.episodes} Durchläufe, seed {args.seed}):")
        for key, value in summary.items():
            print(f"  {key:22s} " + ("–" if value is None else f"{value:.3g}"))

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_rounded(out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nGespeichert: {path}")


if __name__ == "__main__":
    main()
