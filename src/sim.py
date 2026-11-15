"""
Physik-Simulation von Nova in MuJoCo (nur am PC).

Spielt denselben Ablauf wie die Animation in gait.py ab (Stehen, Sitzen, Trab),
aber mit Schwerkraft, Bodenkontakt und Servos mit begrenzter Kraft und
Geschwindigkeit. So sieht man, ob Nova dabei stehen bleibt oder umfällt.

    python src/sim.py              # Fenster mit 3D-Ansicht, Ablauf wiederholt sich
    python src/sim.py --headless   # ohne Fenster, nur Auswertung
    python src/sim.py --headless --speed 100 --period 0.6   # andere Trab-Werte ausprobieren

Braucht `mujoco` und `numpy` im venv. Die Steuerung läuft wie später am Pi mit
50 Hz: alle 20 ms neue Zielwinkel für die 12 Gelenke.

Servo-Modell (Plan Abschnitt 5.1), gerechnet bei jedem Physikschritt (2 ms):
- Zielwinkel auf die Gelenkgrenzen begrenzen (wie Servos.clamp) und auf die
  Stufen des PCA9685 runden (0,44°).
- Kalibrierfehler dazu (der echte Servo steht nie genau dort, wo man denkt).
- Totzeit: Der Servo reagiert erst nach SERVO_DEAD_TIME auf einen neuen Befehl.
- Moment = Kp · Fehler − Kd · Drehzahl, mit Totzone, höchstens das Blockiermoment
  und in Bewegungsrichtung weniger, je schneller der Servo dreht (Kennlinie).
Alle Werte stehen in `ServoParams` und lassen sich pro Durchlauf ändern
(Randomisierung, siehe model/randomization.py).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import mujoco
import numpy as np

from gait import Frame, demo_frames
from kinematics import LegAngles
from nova_model import (
    SERVO_DEAD_TIME, SERVO_DEADBAND, SERVO_KD, SERVO_KP, SERVO_SPEED, SERVO_TORQUE, build_mjcf,
)
from servos import ANGLE_STEP, JOINT_ORDER, JOINTS, Servos

CONTROL_HZ = 50.0


# ---------------------------------------------------------------------------
# Servo-Modell
# ---------------------------------------------------------------------------

@dataclass
class ServoParams:
    """Eigenschaften der Servos. Zahlen gelten für alle 12, Arrays (12 Werte) je Gelenk."""
    stall_torque: float = SERVO_TORQUE   # N·m, Blockiermoment
    max_speed: float = SERVO_SPEED       # rad/s ohne Last
    kp: float = SERVO_KP                 # N·m/rad
    kd: float = SERVO_KD                 # N·m·s/rad
    deadband: float = SERVO_DEADBAND     # rad
    dead_time: float = SERVO_DEAD_TIME   # s
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(len(JOINT_ORDER)))  # rad, Kalibrierfehler


def servo_torque(error: np.ndarray, speed: np.ndarray, p: ServoParams) -> np.ndarray:
    """Moment der Servos in N·m. error = Ziel − Ist (rad), speed = Drehzahl (rad/s)."""
    # (np.minimum/np.maximum statt np.clip, weil das bei 12 Werten deutlich schneller ist.
    #  Die Funktion läuft 500-mal pro simulierter Sekunde.)
    # Totzone: Kleine Fehler sieht der Servo nicht
    error = np.copysign(np.maximum(np.abs(error) - p.deadband, 0.0), error)
    torque = p.kp * error - p.kd * speed
    # Kennlinie: In Drehrichtung fällt das Moment mit der Drehzahl, bei max_speed ist es 0.
    # Gegen die Drehrichtung (Bremsen, Halten) bleibt das volle Blockiermoment.
    s = speed / p.max_speed
    upper = p.stall_torque * np.minimum(np.maximum(1.0 - s, 0.0), 1.0)
    lower = -p.stall_torque * np.minimum(np.maximum(1.0 + s, 0.0), 1.0)
    return np.minimum(np.maximum(torque, lower), upper)


_ZERO = np.array([JOINTS[name].zero for name in JOINT_ORDER])


def command_angles(angles: np.ndarray) -> np.ndarray:
    """Was beim Servo ankommt: Kinematik-Winkel (Grad, 12 Stück) begrenzt und auf PCA9685-Stufen gerundet.

    Begrenzt wird mit Servos.clamp(), genau wie am echten Hund (NaN -> Grundstellung).
    """
    servo = np.array([Servos.clamp(name, a) for name, a in zip(JOINT_ORDER, _ZERO + angles)])
    return np.round(servo / ANGLE_STEP) * ANGLE_STEP - _ZERO


class TorsoState(NamedTuple):
    """Lage des Rumpfs: Position in mm, Winkel in Grad wie bei BodyPose."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def _index(addresses: list[int]) -> slice | np.ndarray:
    """Adressen der Gelenke in qpos/qvel. Liegen sie hintereinander, geht ein Bereich (schneller)."""
    if addresses == list(range(addresses[0], addresses[0] + len(addresses))):
        return slice(addresses[0], addresses[0] + len(addresses))
    return np.array(addresses)


class NovaSim:
    def __init__(self, params: ServoParams | None = None, self_collision: bool = False) -> None:
        self.params = params or ServoParams()  # darf jederzeit ausgetauscht werden
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self_collision))
        self.data = mujoco.MjData(self.model)
        self.torso_id = self.model.body("torso").id
        self._qpos_adr = _index([self.model.joint(name).qposadr[0] for name in JOINT_ORDER])
        self._qvel_adr = _index([self.model.joint(name).dofadr[0] for name in JOINT_ORDER])
        self._queue: deque[tuple[int, np.ndarray]] = deque()  # (ab Physikschritt, Ziel) wegen der Totzeit
        self._target = np.zeros(len(JOINT_ORDER))  # rad, worauf die Servos gerade regeln
        self._steps = 0       # Physikschritte seit reset()
        self._push_steps = 0  # wie lange der Stoß noch dauert
        # Lagesensor: Werte der letzten Physikschritte, damit er verzögert messen kann
        gyro, accel = self.model.sensor("gyro").adr[0], self.model.sensor("accel").adr[0]
        self._imu_adr = _index([gyro, gyro + 1, gyro + 2, accel, accel + 1, accel + 2])
        self._imu_log: deque[np.ndarray] = deque(maxlen=32)

    @staticmethod
    def joint_angles(angles: dict[str, LegAngles]) -> np.ndarray:
        """Kinematik-Winkel aller Beine -> 12 Werte in Grad, Reihenfolge JOINT_ORDER."""
        return np.array([getattr(angles[name[:2]], name[3:]) for name in JOINT_ORDER], dtype=float)

    def _servo_targets(self, angles: dict[str, LegAngles]) -> np.ndarray:
        """Wohin die Servos wirklich fahren, in rad: begrenzt, gerundet, mit Kalibrierfehler."""
        return np.radians(command_angles(self.joint_angles(angles))) + self.params.offsets

    def reset(self, angles: dict[str, LegAngles], height: float) -> None:
        """Hund direkt in eine Pose stellen, Körpermitte auf `height` mm."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = height / 1000
        self._target = self._servo_targets(angles)
        self.data.qpos[self._qpos_adr] = self._target
        self._queue.clear()
        self._steps = 0
        self._push_steps = 0
        mujoco.mj_forward(self.model, self.data)
        self._imu_log.clear()
        self._imu_log.append(self.data.sensordata[self._imu_adr].copy())

    def step(self, angles: dict[str, LegAngles], period: float | None = None) -> None:
        """Neue Zielwinkel schicken und eine Steuerperiode (Standard 20 ms) Physik rechnen."""
        period = 1.0 / CONTROL_HZ if period is None else period
        dt = self.model.opt.timestep
        self._queue.append((self._steps + round(self.params.dead_time / dt), self._servo_targets(angles)))

        for _ in range(round(period / dt)):
            while self._queue and self._queue[0][0] <= self._steps:
                self._target = self._queue.popleft()[1]
            error = self._target - self.data.qpos[self._qpos_adr]
            self.data.ctrl[:] = servo_torque(error, self.data.qvel[self._qvel_adr], self.params)
            mujoco.mj_step(self.model, self.data)
            self._imu_log.append(self.data.sensordata[self._imu_adr].copy())
            self._steps += 1
            if self._push_steps > 0:
                self._push_steps -= 1
                if self._push_steps == 0:
                    self.data.xfrc_applied[self.torso_id] = 0.0

    def push(self, force: tuple[float, float, float], duration: float) -> None:
        """Stoß auf die Rumpfmitte: Kraft in N (Welt: x vorne, y links, z oben) für `duration` s."""
        self.data.xfrc_applied[self.torso_id, :3] = force
        self._push_steps = max(1, round(duration / self.model.opt.timestep))

    def torso(self) -> TorsoState:
        x, y, z = (v * 1000 for v in self.data.xpos[self.torso_id])
        r = self.data.xmat[self.torso_id]  # Drehmatrix, zeilenweise
        return TorsoState(
            x, y, z,
            roll=math.degrees(math.atan2(r[7], r[8])),
            pitch=math.degrees(math.asin(max(-1.0, min(1.0, -r[6])))),
            yaw=math.degrees(math.atan2(r[3], r[0])),
        )

    def torques(self) -> list[float]:
        """Aktuelles Drehmoment der 12 Servos in N·m (Reihenfolge JOINT_ORDER)."""
        return list(self.data.actuator_force)

    def imu(self, delay: float = 0.0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Rohwerte des Lagesensors wie beim MPU6050: (Drehrate in rad/s, Beschleunigung in m/s²).

        delay (s): Wert von vor so langer Zeit (in 2-ms-Schritten, höchstens ca. 60 ms).
        """
        back = round(delay / self.model.opt.timestep)
        sample = self._imu_log[max(0, len(self._imu_log) - 1 - back)]
        return tuple(sample[:3].tolist()), tuple(sample[3:].tolist())

    def distance(self) -> float | None:
        """Abstand nach vorne in mm, None wenn nichts im Weg ist."""
        d = float(self.data.sensor("range").data[0])
        return None if d < 0 else d * 1000


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

def run_headless(frames: list[Frame], self_collision: bool = False) -> None:
    sim = NovaSim(self_collision=self_collision)
    sim.reset(frames[0].angles, frames[0].pose.body.height)

    worst = {"Höhe [mm]": (0.0, ""), "Roll [°]": (0.0, ""), "Pitch [°]": (0.0, "")}
    peak = {"shoulder": 0.0, "upper": 0.0, "lower": 0.0}
    fallen = None

    for frame in frames:
        sim.step(frame.angles)
        state = sim.torso()
        plan = frame.pose.body
        for key, error in (
            ("Höhe [mm]", state.z - plan.height),
            ("Roll [°]", state.roll - plan.roll),
            ("Pitch [°]", state.pitch - plan.pitch),
        ):
            if abs(error) > abs(worst[key][0]):
                worst[key] = (error, f"{frame.text}, t = {frame.time:.1f} s")
        for name, torque in zip(JOINT_ORDER, sim.torques()):
            peak[name[3:]] = max(peak[name[3:]], abs(torque))
        if fallen is None and (state.z < 60 or abs(state.roll) > 45 or abs(state.pitch - plan.pitch) > 45):
            fallen = f"{frame.text}, t = {frame.time:.1f} s"

    end = sim.torso()
    planned = frames[-1].distance
    print("Ergebnis der Simulation")
    print(f"  Umgefallen:         {'JA, bei ' + fallen if fallen else 'nein'}")
    print(f"  Gelaufen:           {end.x:.0f} mm (geplant {planned:.0f} mm, {100 * end.x / planned:.0f} %)")
    print(f"  Seitlich abgedriftet: {end.y:+.0f} mm, gedreht: {end.yaw:+.1f}°")
    print("  Größte Abweichung von der geplanten Körperlage:")
    for key, (error, where) in worst.items():
        print(f"    {key:10s} {error:+6.1f}  ({where})")
    print(f"  Höchstes Servo-Drehmoment (Grenze {SERVO_TORQUE:.1f} N·m):")
    for joint, torque in peak.items():
        print(f"    {joint:8s} {torque:.2f} N·m ({100 * torque / SERVO_TORQUE:.0f} %)")


def run_viewer(frames: list[Frame], self_collision: bool = False) -> None:
    import mujoco.viewer

    sim = NovaSim(self_collision=self_collision)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.torso_id
        viewer.cam.distance = 0.9
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 120
        while viewer.is_running():
            sim.reset(frames[0].angles, frames[0].pose.body.height)
            for frame in frames:
                start = time.perf_counter()
                sim.step(frame.angles)
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nova in MuJoCo: Stehen, Sitzen, Trab")
    parser.add_argument("--headless", action="store_true", help="ohne Fenster, nur Auswertung ausgeben")
    parser.add_argument("--speed", type=float, default=150.0, help="Trab-Geschwindigkeit in mm/s")
    parser.add_argument("--period", type=float, default=0.5, help="Trab-Periode in s")
    parser.add_argument("--step-height", type=float, default=25.0, help="Schritthöhe in mm")
    parser.add_argument("--duty", type=float, default=0.6, help="Anteil der Periode, den ein Fuß am Boden steht")
    parser.add_argument("--self-collision", action="store_true", help="Beine können aneinander anstoßen (langsamer)")
    args = parser.parse_args()

    demo = demo_frames(CONTROL_HZ, args.speed, args.period, args.step_height, args.duty)
    if args.headless:
        run_headless(demo, args.self_collision)
    else:
        run_viewer(demo, args.self_collision)
