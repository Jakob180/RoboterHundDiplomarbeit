"""
Trainingsumgebung für Nova im Gymnasium-Format (Plan Abschnitt 4.1 und 4.2).

Gymnasium ist der Standard für solche Umgebungen. Trainingsprogramme wie
Stable-Baselines3 (Phase 3) können damit direkt arbeiten:

    env = NovaEnv(mode="residual")            # Stufe 1; "full" = Stufe 2
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(aktion)

Ein Schritt = 20 ms: Netz-Ausgabe -> policy.py (Trab + Korrektur bzw. Gelenkwinkel)
-> Servo-Modell und Physik -> Lagesensor mit Fehlern -> Filter -> neue Beobachtung.
Die Belohnung rechnet mit Werten, die es nur in der Simulation gibt (echtes
Tempo, echte Neigung); das Netz selbst sieht nur, was der Pi auch hat.

Bei jedem reset() wird ein neuer Hund gewürfelt (randomization.py), wenn
randomize=True. Danach steht er 0,5 s, bevor das Netz übernimmt. Für faire
Vergleiche lassen sich Hund und Befehl fest vorgeben:
    env.reset(seed=0, options={"randomization": r, "command": (150.0, 0.0)})
Am Ende einer Episode steht in info["nova"] die gelaufene Strecke (Anteil des
Befehls) und ob Nova umgefallen ist.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from gait import STAND  # noqa: E402
from imu import ComplementaryFilter  # noqa: E402
from kinematics import LEGS  # noqa: E402
from nova_model import FOOT_RADIUS  # noqa: E402
from policy import FOOT_SCALE, N_ACTIONS, OBS_CLIP, STAND_ANGLES, FullMode, ResidualMode, joint_dict  # noqa: E402
from randomization import (  # noqa: E402
    PUSH_FORCE, PushStyle, Randomization, apply, imu_reading, sample, sample_pushes,
)
from sim import CONTROL_HZ, NovaSim  # noqa: E402

DT = 1.0 / CONTROL_HZ


def course_error(actual: float, planned: float) -> float:
    """Abweichung der Blickrichtung vom Kurs in Grad, -180..180 (+ = zu weit nach links)."""
    return (actual - planned + 180.0) % 360.0 - 180.0


class NovaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str = "residual",
        randomize: bool = True,
        pushes: bool = True,
        max_yaw_rate: float = config.COMMAND_YAW_RATE,
        episode_seconds: float = config.EPISODE_SECONDS,
        push_force: float = PUSH_FORCE[1],
        foot_scale=FOOT_SCALE,
        push_style: PushStyle | None = None,
        self_collision: bool | None = None,
    ) -> None:
        if mode not in config.REWARD:
            raise ValueError(f"mode muss 'residual' oder 'full' sein, nicht {mode!r}")
        self.mode = mode
        self.randomize = randomize      # neuer zufälliger Hund bei jedem reset()
        self.pushes = pushes            # Stöße alle 3 bis 6 s
        self.push_force = push_force    # N, stärkster Stoß
        self.push_style = push_style    # gezieltes Stoßtraining (randomization.PUSH_STYLES)
        self.max_yaw_rate = max_yaw_rate
        self.episode_steps = round(episode_seconds * CONTROL_HZ)
        self.weights = dict(config.REWARD[mode])  # eigene Kopie, z. B. für das Ausblenden der Nachahmung
        # Stufe 2: Beine können aneinander und am Rumpf anstoßen (Plan 4.2), Stufe 1 wie bisher
        self.self_collision = mode == "full" if self_collision is None else self_collision
        self.policy = ResidualMode(foot_scale) if mode == "residual" else FullMode()
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, (self.policy.n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_ACTIONS,), np.float32)
        self.sim: NovaSim | None = None

    # -----------------------------------------------------------------------
    # Gymnasium
    # -----------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        total = self.episode_steps * DT + config.WARMUP_SECONDS
        if "randomization" in options:      # fest vorgegebener Hund (Vergleiche)
            self.randomization = options["randomization"]
        elif self.randomize:
            self.randomization = sample(rng, self.pushes, total, self.push_force, self.push_style)
        else:
            self.randomization = Randomization(
                pushes=sample_pushes(rng, total, self.push_force, self.push_style) if self.pushes else [],
            )
        self._fixed_command = options.get("command")  # (mm/s, °/s) für die ganze Episode

        self.sim = NovaSim(self_collision=self.self_collision)
        apply(self.sim, self.randomization)
        stand = joint_dict(STAND_ANGLES)
        self.sim.reset(stand, STAND.body.height)
        self._frame = self.sim.model.geom("frame").id
        self._floor = self.sim.model.geom("floor").id
        self._feet = np.array([self.sim.model.geom(f"{leg}_foot").id for leg in LEGS])
        self._pushes = sorted(self.randomization.pushes)
        self.policy.reset()
        self.filter = ComplementaryFilter()
        self._gyro = (0.0, 0.0, 0.0)  # zuletzt gemessene Drehrate (rad/s)
        for _ in range(round(config.WARMUP_SECONDS * CONTROL_HZ)):
            self._advance(stand)
        self._start_yaw = self.sim.torso().yaw  # Grad, Richtung beim Start = Kurs 0
        self._feet_before = self.sim.data.geom_xpos[self._feet, :2].copy()

        self.steps = 0
        self.walked = 0.0      # mm, in Blickrichtung gelaufen
        self.commanded = 0.0   # mm, so weit hätte es laut Befehl sein sollen
        self._next_command = 0.0
        self._update_command()
        return self.policy.observe(self.filter.up, self._gyro), {}

    def step(self, action):
        previous = self.policy.last_action.copy()
        angles = self.policy.act(action, DT)
        start = self.sim.data.qpos[:3].copy()
        dt = self._advance(angles)
        self.steps += 1

        terms, fallen = self._rate(self.policy.last_action, previous, start, dt)
        reward = sum(w * terms[k] for k, w in self.weights.items())
        truncated = not fallen and self.steps >= self.episode_steps
        self._update_command()
        obs = self.policy.observe(self.filter.up, self._gyro)
        info = {"terms": terms}
        if fallen or truncated:
            share = self.walked / self.commanded if self.commanded > 0 else 0.0
            info["nova"] = {"strecke": share, "umgefallen": float(fallen)}
        return obs, float(reward), fallen, truncated, info

    def set_reward_weight(self, name: str, value: float) -> None:
        """Gewicht eines Belohnungs-Terms ändern (nur in dieser Umgebung), z. B. Nachahmung ausblenden."""
        if name not in self.weights:
            raise KeyError(f"Belohnungs-Term {name!r} gibt es in Stufe {self.mode!r} nicht")
        self.weights[name] = float(value)

    @property
    def last_imu(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Zuletzt gelesener Lagesensor: (Richtung oben aus dem Filter, Drehrate in rad/s).

        Daraus ist auch die letzte Beobachtung gebaut, z. B. für den Lehrer in model/imitation.py.
        """
        return tuple(float(v) for v in self.filter.up), tuple(float(v) for v in self._gyro)

    # -----------------------------------------------------------------------
    # Innenleben
    # -----------------------------------------------------------------------

    def _advance(self, angles) -> float:
        """20 ms (± Zittern des Steuertakts) Physik, dann Lagesensor lesen. Gibt die echte Dauer zurück."""
        sim, r = self.sim, self.randomization
        while self._pushes and self._pushes[0].time <= sim.data.time + 1e-9:
            push = self._pushes.pop(0)
            sim.push(push.force, push.duration)
        before = sim.data.time
        jitter = self.np_random.uniform(-r.period_jitter, r.period_jitter) if r.period_jitter else 0.0
        sim.step(angles, DT + jitter)
        dt = sim.data.time - before
        gyro, accel = imu_reading(sim, r.imu, self.np_random)
        self.filter.update(gyro, accel, dt)
        self._gyro = gyro
        return dt

    def _update_command(self) -> None:
        """Alle 2 bis 4 s neues Tempo (und, wenn eingeschaltet, neue Drehrate)."""
        if self._fixed_command is not None:
            self.policy.set_command(*self._fixed_command)
            return
        if self.steps * DT < self._next_command - 1e-9:
            return
        rng = self.np_random
        speed = rng.uniform(*config.COMMAND_SPEED)
        yaw_rate = rng.uniform(-self.max_yaw_rate, self.max_yaw_rate) if self.max_yaw_rate else 0.0
        self.policy.set_command(speed, yaw_rate)
        self._next_command += rng.uniform(*config.COMMAND_EVERY)

    def _feet_terms(self, dt: float) -> tuple[float, float]:
        """(Füße rutschen, Füße zu tief im Schwung) für diesen Schritt."""
        data = self.sim.data
        pos = data.geom_xpos[self._feet]
        g1, g2 = data.contact.geom1, data.contact.geom2
        speed = np.linalg.norm(pos[:, :2] - self._feet_before, axis=1) / dt   # m/s waagrecht
        self._feet_before = pos[:, :2].copy()
        slip = 0.0
        for k, foot in enumerate(self._feet):
            if np.any(((g1 == foot) & (g2 == self._floor)) | ((g1 == self._floor) & (g2 == foot))):
                slip += float(speed[k])
        low = 0.0
        a, b = config.CLEARANCE_WINDOW
        for k, leg in enumerate(LEGS):
            u = self.policy.trot.swing(leg)
            if u is not None and a <= u <= b:
                height = pos[k, 2] * 1000 - FOOT_RADIUS                     # mm über dem Boden
                low += max(0.0, (config.CLEARANCE - height) / config.CLEARANCE)
        return slip, low

    def _rate(self, action: np.ndarray, previous: np.ndarray, start: np.ndarray, dt: float):
        """Belohnungs-Terme dieses Schritts (Plan-Tabelle 4.1) und ob Nova umgefallen ist."""
        data = self.sim.data
        rot = data.xmat[self.sim.torso_id].reshape(3, 3)
        up = rot[2]                                         # Richtung "oben" in Rumpf-Koordinaten
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        moved = (data.qpos[:3] - start) / dt                # m/s, Welt
        vx = math.cos(yaw) * moved[0] + math.sin(yaw) * moved[1]
        vy = -math.sin(yaw) * moved[0] + math.cos(yaw) * moved[1]
        vz = moved[2]
        spin = data.qvel[3:6]                               # rad/s, Rumpf-Koordinaten
        load = data.actuator_force / self.randomization.servo.stall_torque
        self.walked += vx * dt * 1000
        self.commanded += self.policy.speed * dt

        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
        off_course = course_error(math.degrees(yaw) - self._start_yaw, self.policy.trot.heading)
        # Rumpf berührt den Boden? (als Arrays, viel schneller als Kontakt für Kontakt)
        g1, g2 = data.contact.geom1, data.contact.geom2
        on_floor = bool(np.any(((g1 == self._frame) & (g2 == self._floor)) | ((g1 == self._floor) & (g2 == self._frame))))
        fallen = data.qpos[2] * 1000 < config.FALL_HEIGHT or tilt > config.FALL_TILT or on_floor

        imitation = np.radians(self.policy.angles - getattr(self.policy, "reference", self.policy.angles))
        slip, low = self._feet_terms(dt)
        terms = {
            "tempo": math.exp(-((vx - self.policy.speed / 1000) / config.SPEED_SIGMA) ** 2),
            "drehrate": math.exp(-((spin[2] - math.radians(self.policy.yaw_rate)) / config.YAW_SIGMA) ** 2),
            "kurs": math.exp(-(off_course / config.COURSE_SIGMA) ** 2),
            "rutschen": vy ** 2,
            "neigung": up[0] ** 2 + up[1] ** 2,
            "wackeln": spin[0] ** 2 + spin[1] ** 2,
            "huepfen": vz ** 2,
            "korrektur": float(np.sum(action ** 2)) / N_ACTIONS,
            "ruckeln": float(np.sum((action - previous) ** 2)) / N_ACTIONS,
            "servo_last": float(np.mean(load ** 2)),
            "anschlag": float(np.mean(np.abs(load) > config.TORQUE_LIMIT)),
            "nachahmung": math.exp(-float(np.sum(imitation ** 2)) / config.IMITATION_SIGMA ** 2),
            "fuss_rutschen": slip,
            "fusshoehe": low,
            "umfallen": 1.0 if fallen else 0.0,
        }
        return terms, fallen
