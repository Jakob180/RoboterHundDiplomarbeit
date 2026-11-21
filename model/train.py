"""
PPO-Training für Nova (Plan Phase 3, Abschnitte 4.1 und 5.3). Nur am PC.

    python model/train.py                        # Stufe 1, ganzer Lehrplan (ca. 15 Mio. Schritte)
    python model/train.py --quick                # Probelauf in 1 bis 2 Minuten: läuft alles?
    python model/train.py --envs 8 --scale 0.5   # 8 Hunde parallel, halb so viele Schritte
    tensorboard --logdir runs                    # Lernkurven im Browser: http://localhost:6006
    python model/train.py --push-training checkpoints/<lauf>/final.zip   # fertiges Netz mit starken Seitenstößen weitertrainieren
    python model/train.py --start-from checkpoints/full-bc/final.zip     # Stufe 2: PPO-Feinschliff des nachgemachten Netzes

Lehrplan aus config.CURRICULUM: erst ohne Zufall, dann mit Randomisierung, dann
mit Stößen. Alle Abschnitte landen in einer Lernkurve in runs/<lauf>/.
Zwischenstände und Endergebnis in checkpoints/<lauf>/ (final.zip), danach:
    python model/export_onnx.py checkpoints/<lauf>/final.zip

Gerechnet wird nur auf der CPU (bei so kleinen Netzen bringt die Grafikkarte
nichts). Es laufen so viele Hunde parallel wie die CPU Threads hat; jeder
braucht ca. 300 MB Arbeitsspeicher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import policy  # noqa: E402
from nova_env import NovaEnv  # noqa: E402
from policy import FOOT_SCALE  # noqa: E402
from randomization import PUSH_STYLES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_env(mode: str, randomize: bool, push_force, seed: int, foot_scale=FOOT_SCALE) -> NovaEnv:
    """Eine Umgebung (steht oben in der Datei, damit Windows sie an Unterprozesse geben kann).

    push_force: stärkster Stoß in N (0 = keine Stöße) oder Name einer Stoß-Art aus
    randomization.PUSH_STYLES. foot_scale: Fußkorrektur in mm (Stufe 1).
    """
    if isinstance(push_force, str):
        env = NovaEnv(mode=mode, randomize=randomize, pushes=True, push_style=PUSH_STYLES[push_force],
                      foot_scale=foot_scale)
    elif push_force > 0:
        env = NovaEnv(mode=mode, randomize=randomize, pushes=True, push_force=push_force, foot_scale=foot_scale)
    else:
        env = NovaEnv(mode=mode, randomize=randomize, pushes=False, foot_scale=foot_scale)
    env.reset(seed=seed)
    return env


def actor_parameters(policy) -> list:
    """Alles, was zum Actor gehört (die Ausgabe des Netzes), ohne den Critic."""
    return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(), policy.log_std]


def freeze_actor(policy, frozen: bool) -> None:
    """Actor einfrieren (nur der Critic lernt) oder wieder freigeben."""
    for parameter in actor_parameters(policy):
        parameter.requires_grad_(not frozen)


def set_exploration(policy, log_std: float) -> None:
    """Zufall beim Ausprobieren neu setzen (Standardabweichung e^log_std auf jede Ausgabe)."""
    import torch

    with torch.no_grad():
        policy.log_std.fill_(log_std)


def imitation_weight(done: float, total: float) -> float:
    """Gewicht der Nachahmung: linear von IMITATION_FADE[0] auf [1] über `total` Schritte."""
    start, end = config.IMITATION_FADE
    share = min(max(done / total, 0.0), 1.0) if total > 0 else 1.0
    return start + (end - start) * share


class ImitationFade(BaseCallback):
    """Stufe 2: Nachahmung im Lauf des Feinschliffs ausblenden, damit das Netz besser werden darf als der Trab."""

    def __init__(self, start: int, total: int) -> None:
        super().__init__()
        self.start, self.total = start, total

    def _on_rollout_start(self) -> None:
        weight = imitation_weight(self.num_timesteps - self.start, self.total)
        self.training_env.env_method("set_reward_weight", "nachahmung", weight)
        self.logger.record("nova/nachahmung", weight)

    def _on_step(self) -> bool:
        return True


class NovaLogger(BaseCallback):
    """Schreibt die Belohnungs-Terme und Kennzahlen der Episoden nach TensorBoard."""

    def _on_rollout_start(self) -> None:
        self._terms: dict[str, float] = {}
        self._count = 0
        self._episodes: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for key, value in info.get("terms", {}).items():
                self._terms[key] = self._terms.get(key, 0.0) + value
            self._count += 1
            if "nova" in info:
                self._episodes.append(info["nova"])
        return True

    def _on_rollout_end(self) -> None:
        for key, total in self._terms.items():
            self.logger.record(f"terme/{key}", total / max(self._count, 1))
        if self._episodes:
            self.logger.record("nova/strecke", float(np.mean([e["strecke"] for e in self._episodes])))
            self.logger.record("nova/umgefallen", float(np.mean([e["umgefallen"] for e in self._episodes])))


def train(
    mode: str = "residual",
    run: str | None = None,
    envs: int | None = None,
    scale: float = 1.0,
    quick: bool = False,
    seed: int = 0,
    out: Path = ROOT,
    vec_cls=SubprocVecEnv,
    foot_scale=FOOT_SCALE,
    start_from: Path | None = None,
) -> Path:
    """Ganzen Lehrplan trainieren. Gibt den Pfad zu final.zip zurück.

    start_from: fertiges Netz (final.zip mit settings.json daneben), das weitertrainiert
    wird. Modus und Fußkorrektur kommen dann vom Netz: Ein Netz aus Stufe 1 bekommt das
    Stoßtraining (config.PUSH_TRAINING), ein nachgemachtes Netz aus model/imitation.py den
    PPO-Feinschliff von Stufe 2 (config.CURRICULUM["full"]).
    """
    source_settings = None
    if start_from is not None:
        start_from = Path(start_from)
        source_settings = json.loads((start_from.parent / "settings.json").read_text(encoding="utf-8"))
        policy.check_settings(source_settings, start_from.name)
        mode, foot_scale = source_settings["mode"], source_settings["scales"]["foot"]
        stages = config.PUSH_TRAINING if mode == "residual" else config.CURRICULUM[mode]
    elif mode == "full":
        raise ValueError("Stufe 2 startet mit einem nachgemachten Netz: zuerst model/imitation.py, "
                         "dann train.py --start-from checkpoints/<lauf>/final.zip")
    elif mode in config.CURRICULUM:
        stages = config.CURRICULUM[mode]
    else:
        raise ValueError(f"Unbekannter mode={mode!r}, erwartet 'residual' oder 'full'")
    settings = dict(config.PPO_SETTINGS)
    n_envs = envs or os.cpu_count() or 1
    if quick:  # nur prüfen, ob alles läuft
        stages = [stage._replace(millions=0.001) for stage in stages]
        settings.update(n_steps=64, batch_size=64, n_epochs=2)
        n_envs = min(n_envs, 2)

    prefix = f"{mode}-stoss" if start_from is not None and mode == "residual" else mode
    run = run or f"{prefix}-{time.strftime('%Y%m%d-%H%M')}"
    folder = Path(out) / "checkpoints" / run
    folder.mkdir(parents=True, exist_ok=True)
    # Einstellungen aus der Zeit des Trainings, gehören später als .json zum Netz
    meta = source_settings or policy.settings(NovaEnv(mode=mode, foot_scale=foot_scale).policy)
    (folder / "settings.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    model = None
    warmup = sum(int(st.millions * 1e6 * scale) for st in stages if not st.train_actor)
    fade_total = sum(int(st.millions * 1e6 * scale) for st in stages if st.train_actor)
    for i, stage in enumerate(stages):
        name, millions, randomize, push_force = stage.name, stage.millions, stage.randomize, stage.push
        steps = int(millions * 1e6 * scale)
        vec = VecMonitor(vec_cls([
            partial(make_env, mode, randomize, push_force, seed + 1000 * i + k, tuple(foot_scale))
            for k in range(n_envs)
        ]))
        if model is None and start_from is not None:
            print(f"Weitertrainiert wird {start_from}", flush=True)
            model = PPO.load(
                start_from, env=vec, device="cpu", tensorboard_log=str(Path(out) / "runs"),
                custom_objects=settings if quick else None,
            )
            if mode == "full":  # nachgemachtes Netz: wenig Zufall, sonst stolpert es gleich
                set_exploration(model.policy, config.FULL_LOG_STD_INIT)
        elif model is None:
            model = PPO(
                "MlpPolicy", vec, device="cpu", seed=seed, verbose=0,
                tensorboard_log=str(Path(out) / "runs"),
                policy_kwargs={
                    "net_arch": {"pi": config.NET_ARCH, "vf": config.NET_ARCH},
                    "log_std_init": config.LOG_STD_INIT,
                },
                **settings,
            )
        else:
            model.set_env(vec)
        freeze_actor(model.policy, not stage.train_actor)
        if isinstance(push_force, str):
            style = PUSH_STYLES[push_force]
            pushes = f", Seitenstöße {style.force[0]:.0f} bis {style.force[1]:.0f} N"
        else:
            pushes = f", Stöße bis {push_force:.0f} N" if push_force > 0 else ""
        actor = "" if stage.train_actor else ", nur der Critic lernt"
        print(f"[{time.strftime('%H:%M')}] Abschnitt {name}: {steps:,} Schritte, {n_envs} Hunde parallel{pushes}{actor}",
              flush=True)
        callbacks = [
            CheckpointCallback(max(config.CHECKPOINT_EVERY // n_envs, 1), str(folder), name_prefix=name),
            NovaLogger(),
        ]
        if mode == "full":
            callbacks.append(ImitationFade(warmup, fade_total))
        model.learn(
            steps, reset_num_timesteps=(i == 0), tb_log_name=run, callback=callbacks,
        )
        model.save(folder / f"{name}.zip")
        vec.close()

    final = folder / "final.zip"
    model.save(final)
    print(f"[{time.strftime('%H:%M')}] Fertig: {final}", flush=True)
    return final


def next_steps(final: Path) -> str:
    """Befehle für danach, Pfade in Anführungszeichen (Leerzeichen, z. B. "Ki Projekte")."""
    python = r".venv\Scripts\python" if os.name == "nt" else ".venv/bin/python"
    return (
        "Weiter mit:\n"
        f'  {python} model/evaluate.py --policy "{final}"\n'
        f'  {python} model/export_onnx.py "{final}"'
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Netz für Nova trainieren (PPO)")
    parser.add_argument("--mode", default="residual", help="residual = Stufe 1 (Stufe 2 kommt in Phase 4)")
    parser.add_argument("--envs", type=int, help="Hunde parallel, Standard: Zahl der CPU-Threads")
    parser.add_argument("--scale", type=float, default=1.0, help="Faktor auf die Schritte des Lehrplans")
    parser.add_argument("--run", help="Name des Laufs, Standard: Modus + Datum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="kurzer Probelauf")
    parser.add_argument(
        "--foot-scale", type=float, nargs=3, default=list(FOOT_SCALE), metavar=("X", "Y", "Z"),
        help="Stufe 1: größte Fußkorrektur in mm, vorne/seitlich/oben (Standard %(default)s)",
    )
    parser.add_argument(
        "--push-training", metavar="FINAL_ZIP",
        help="fertiges Netz mit starken Seitenstößen weitertrainieren (config.PUSH_TRAINING)",
    )
    parser.add_argument(
        "--start-from", metavar="FINAL_ZIP",
        help="fertiges Netz weitertrainieren: Stufe 1 -> Stoßtraining, Stufe 2 (aus imitation.py) -> PPO-Feinschliff",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    final = train(args.mode, args.run, args.envs, args.scale, args.quick, args.seed, foot_scale=args.foot_scale,
                  start_from=args.start_from or args.push_training)
    print(next_steps(final))


if __name__ == "__main__":  # Pflicht unter Windows, sonst starten die Unterprozesse das Training erneut
    main()
