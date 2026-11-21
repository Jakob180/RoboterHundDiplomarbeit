"""
Stufe 2, Schritte 1 und 2 (Plan 4.2): Lehrer-Daten sammeln und nachmachen.

1. Der Lehrer (klassischer Trab, am besten mit dem Netz aus Stufe 1) läuft in der
   Simulation mit Randomisierung. Gespeichert werden die Beobachtung, wie sie das Netz
   der Stufe 2 sieht, und die Gelenkwinkel des Lehrers als Ausgabe von Stufe 2.
   Ausgeführt wird die Lehrer-Ausgabe mit etwas Rauschen, damit das Netz auch
   Zustände neben dem perfekten Trab kennenlernt (die gespeicherte Ausgabe bleibt die
   saubere des Lehrers).
2. Behavior Cloning: Ein Netz in derselben Form wie der PPO-Actor lernt diese
   Ausgaben nachzumachen (überwachtes Lernen, mittlerer quadratischer Fehler).
   Danach läuft Nova in der Simulation allein mit dem Netz. Gespeichert wird es wie
   ein trainiertes Netz (final.zip + settings.json), also gehen evaluate.py,
   export_onnx.py und später der PPO-Feinschliff damit.

    python model/imitation.py                                   # Lehrer: Trab ohne Netz
    python model/imitation.py --teacher checkpoints/<lauf>/final.zip   # Lehrer: Trab + Netz Stufe 1
    python model/evaluate.py --policy checkpoints/<lauf>/final.zip      # läuft es allein?
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import policy  # noqa: E402
from nova_env import NovaEnv  # noqa: E402
from policy import FOOT_SCALE, ResidualMode, angles_to_full_action  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DT = 1.0 / policy.CONTROL_HZ
SAMPLES = 1_000_000   # Paare aus Beobachtung und Lehrer-Ausgabe (Plan: ca. 1 Mio.)
NOISE = 0.05          # Rauschen auf der ausgeführten Ausgabe (0,05 = 1,5° Ober-/Unterschenkel)
EPOCHS = 30
BATCH = 4096
LEARNING_RATE = 1e-3
VALIDATION = 0.05     # Anteil Prüfdaten, die nicht zum Lernen benutzt werden


class Teacher:
    """Lehrer für Stufe 2: klassischer Trab, optional korrigiert vom Netz aus Stufe 1.

    Läuft im Gleichschritt mit der Umgebung (gleicher Befehl, gleiche Phase) und gibt
    seine Gelenkwinkel als Ausgabe von Stufe 2 zurück.
    """

    def __init__(self, network=None, foot_scale=FOOT_SCALE) -> None:
        self.network = network
        self.mode = ResidualMode(foot_scale)

    def reset(self) -> None:
        self.mode.reset()

    def action(self, up, gyro, speed: float, yaw_rate: float) -> np.ndarray:
        """Nächste Ausgabe für Stufe 2 aus Lagesensor und befohlenem Tempo/Drehrate."""
        self.mode.set_command(speed, yaw_rate)
        obs = self.mode.observe(up, gyro)
        correction = np.zeros(policy.N_ACTIONS) if self.network is None else self.network(obs)
        self.mode.act(correction, DT)
        return angles_to_full_action(self.mode.angles)


def load_teacher(path) -> Teacher:
    """Lehrer mit einem fertigen Netz aus Stufe 1 (.zip aus dem Training oder .onnx)."""
    from evaluate import load_network  # erst hier: braucht stable-baselines3 bzw. onnxruntime

    network, meta = load_network(path)
    if meta["mode"] != "residual":
        raise ValueError(f"Der Lehrer muss ein Netz aus Stufe 1 sein, {path} ist {meta['mode']!r}")
    return Teacher(network, meta["scales"]["foot"])


def _collect_part(job: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Ein Teil der Daten (steht oben in der Datei, damit Windows ihn an Unterprozesse geben kann)."""
    samples, seed, teacher_path, noise = job
    teacher = load_teacher(teacher_path) if teacher_path else Teacher()
    env = NovaEnv(mode="full")   # randomisiert, mit Stößen bis 15 N
    rng = np.random.default_rng(seed)
    obs_out = np.empty((samples, env.policy.n_obs), dtype=np.float32)
    act_out = np.empty((samples, policy.N_ACTIONS), dtype=np.float32)
    n = 0
    while n < samples:
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        teacher.reset()
        while n < samples:
            up, gyro = env.last_imu
            a = teacher.action(up, gyro, env.policy.target_speed, env.policy.target_yaw_rate)
            obs_out[n], act_out[n] = obs, a
            n += 1
            executed = np.clip(a + noise * rng.standard_normal(a.shape), -1.0, 1.0)
            obs, _, fallen, truncated, _ = env.step(executed)
            if fallen or truncated:
                break
    return obs_out, act_out


def collect(
    samples: int = SAMPLES, seed: int = 0, workers: int = 1, teacher_path=None, noise: float = NOISE,
) -> tuple[np.ndarray, np.ndarray]:
    """Lehrer laufen lassen. Gibt (Beobachtungen, Lehrer-Ausgaben) zurück, gleicher seed = gleiche Daten."""
    workers = max(1, min(workers, samples))
    share = math.ceil(samples / workers)
    jobs = [(min(share, samples - k * share), (seed, k), teacher_path, noise) for k in range(workers)]
    jobs = [job for job in jobs if job[0] > 0]
    if len(jobs) == 1:
        parts = [_collect_part(jobs[0])]
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_collect_part, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def behavior_cloning(
    obs: np.ndarray, actions: np.ndarray, folder, epochs: int = EPOCHS, batch: int = BATCH,
    learning_rate: float = LEARNING_RATE, seed: int = 0, log=None,
) -> tuple[Path, list[float]]:
    """Actor von PPO (gleiche Form wie im Training) auf die Lehrer-Ausgaben trainieren.

    Speichert folder/final.zip und folder/settings.json. Gibt (Pfad, Fehler auf den
    Prüfdaten vor dem Lernen und nach jeder Runde) zurück.
    """
    import torch
    from stable_baselines3 import PPO

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    with warnings.catch_warnings():  # Hinweis zur batch_size gilt erst beim PPO-Training mit allen Hunden
        warnings.filterwarnings("ignore", message="You have specified a mini-batch size")
        model = PPO(
            "MlpPolicy", NovaEnv(mode="full", randomize=False, pushes=False), device="cpu", seed=seed, verbose=0,
            policy_kwargs={"net_arch": {"pi": config.NET_ARCH, "vf": config.NET_ARCH},
                           "log_std_init": config.FULL_LOG_STD_INIT},
            **config.PPO_SETTINGS,
        )
    p = model.policy

    def actor(x):
        return p.action_net(p.mlp_extractor.forward_actor(p.pi_features_extractor(x)))

    order = np.random.default_rng(seed).permutation(len(obs))
    n_val = max(1, int(len(obs) * VALIDATION))
    x_all, y_all = torch.as_tensor(obs), torch.as_tensor(actions)
    val, train = order[:n_val], order[n_val:]
    params = list(p.mlp_extractor.policy_net.parameters()) + list(p.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    def validation_loss() -> float:
        with torch.no_grad():
            return float(torch.mean((actor(x_all[val]) - y_all[val]) ** 2))

    history = [validation_loss()]
    shuffle = np.random.default_rng(seed + 1)
    for epoch in range(epochs):
        perm = train[shuffle.permutation(len(train))]  # jede Runde neu gemischt
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            loss = torch.mean((actor(x_all[idx]) - y_all[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(validation_loss())
        if log:
            log(f"  Runde {epoch + 1:3d}/{epochs}: Fehler {history[-1]:.5f}")

    final = folder / "final.zip"
    model.save(final)
    (folder / "settings.json").write_text(json.dumps(policy.settings(policy.FullMode()), indent=2) + "\n",
                                          encoding="utf-8")
    return final, history


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stufe 2: Lehrer-Daten sammeln und nachmachen (Behavior Cloning)")
    parser.add_argument("--teacher", help="Netz aus Stufe 1 als Lehrer (final.zip oder .onnx), sonst Trab allein")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Paare aus Beobachtung und Lehrer-Ausgabe")
    parser.add_argument("--workers", type=int, help="Prozesse parallel, Standard: Zahl der CPU-Threads")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Lernrunden über alle Daten")
    parser.add_argument("--noise", type=float, default=NOISE, help="Rauschen auf der ausgeführten Ausgabe")
    parser.add_argument("--run", help="Name des Laufs, Standard: full-bc + Datum")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run = args.run or f"full-bc-{time.strftime('%Y%m%d-%H%M')}"
    folder = ROOT / "checkpoints" / run
    folder.mkdir(parents=True, exist_ok=True)
    workers = args.workers or os.cpu_count() or 1
    teacher = args.teacher or "Trab ohne Netz"
    print(f"[{time.strftime('%H:%M')}] Lehrer: {teacher}. Sammle {args.samples:,} Paare mit {workers} Prozessen ...",
          flush=True)
    obs, actions = collect(args.samples, args.seed, workers, args.teacher, args.noise)
    np.savez(folder / "lehrer-daten.npz", obs=obs, actions=actions)
    print(f"[{time.strftime('%H:%M')}] Daten gespeichert: {folder / 'lehrer-daten.npz'}", flush=True)
    print(f"[{time.strftime('%H:%M')}] Nachmachen lernen ({args.epochs} Runden):", flush=True)
    final, history = behavior_cloning(obs, actions, folder, args.epochs, seed=args.seed,
                                      log=lambda text: print(text, flush=True))
    print(f"[{time.strftime('%H:%M')}] Fertig: {final} (Fehler {history[0]:.4f} -> {history[-1]:.5f})")
    python = r".venv\Scripts\python" if os.name == "nt" else ".venv/bin/python"
    print(f'Weiter mit:\n  {python} model/evaluate.py --policy "{final}"')


if __name__ == "__main__":  # Pflicht unter Windows, sonst starten die Unterprozesse alles erneut
    main()
