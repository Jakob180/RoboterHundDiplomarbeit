"""
Domain Randomization für Nova (Plan Abschnitt 5.2).

Niemand weiß genau, wie schwer der echte Hund ist, wie glatt der Boden ist
oder wie stark jeder Servo zieht. Deshalb wird vor jedem Durchlauf neu
gewürfelt. Was in der Simulation über alle diese Varianten funktioniert,
funktioniert hoffentlich auch am echten Hund.

    r = sample(np.random.default_rng(seed))   # würfeln, gleicher seed = gleiche Werte
    sim = NovaSim()
    apply(sim, r)                             # ins Modell schreiben

Fehler des Lagesensors (Rauschen, Nullpunktfehler, Einbauwinkel, Verzögerung)
stehen in `ImuErrors` und wirken über `imu_reading()`. Der klassische Trab
benutzt den Sensor nicht, erst das Netz. Der Boden ist noch flach.

Braucht `src/` im Suchpfad (machen model/evaluate.py und tests/conftest.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import mujoco
import numpy as np

from kinematics import LEGS
from nova_model import FLOOR_FRICTION, SERVO_KD, SERVO_KP
from servos import JOINT_ORDER
from sim import NovaSim, ServoParams

# Bereiche aus dem Plan (Tabelle 5.2). Startwerte, später an Messungen anpassen.
TORSO_MASS = (0.8, 1.2)          # Faktor auf die Rumpfmasse
TORSO_COM = 20.0                 # mm, Schwerpunkt verschoben um bis zu ± in x und y
LEG_MASS = (0.85, 1.15)          # Faktor auf die Beinmassen
FLOOR = (0.4, 1.2)               # Reibwert Boden
FLOOR_SOLREF = (0.01, 0.04)      # s, Weichheit des Fußkontakts (MuJoCo solref)
STALL_TORQUE = (2.0, 2.7)        # N·m
MAX_SPEED = (380.0, 460.0)       # °/s
GAIN = (0.7, 1.3)                # Faktor auf Kp und Kd
DEAD_TIME = (5, 40)              # ms, in 2-ms-Schritten
CALIBRATION = 3.0                # ± Grad je Gelenk
DEADBAND = (0.0, 2.0)            # Grad je Gelenk (Spiel im Getriebe)
PERIOD_JITTER = 0.002            # ± s am Steuertakt von 20 ms
PUSH_EVERY = (3.0, 6.0)          # s zwischen zwei Stößen
PUSH_FORCE = (0.0, 15.0)         # N, waagrecht in zufälliger Richtung
PUSH_DURATION = (0.1, 0.2)       # s
# Lagesensor (MPU6050). PLATZHALTER, mit dem echten Sensor am Pi nachmessen
IMU_GYRO_NOISE = (0.0, 0.02)     # rad/s Standardabweichung (dazu Vibrationen)
IMU_ACCEL_NOISE = (0.0, 0.2)     # m/s² Standardabweichung
IMU_GYRO_BIAS = 0.02             # ± rad/s je Achse, Rest nach dem Nullpunkt-Abgleich beim Start
IMU_MOUNT = 2.0                  # ± Grad je Achse, Sensor schief eingebaut
IMU_DELAY = (0, 10)              # ms, in 2-ms-Schritten

NOMINAL_SOLREF = 0.02            # MuJoCo-Standard


class Push(NamedTuple):
    time: float                          # s ab Start
    force: tuple[float, float, float]    # N in Weltkoordinaten
    duration: float                      # s


@dataclass(frozen=True)
class ImuErrors:
    """Fehler des Lagesensors für einen Durchlauf. Standardwerte = perfekter Sensor."""
    gyro_noise: float = 0.0                              # rad/s
    accel_noise: float = 0.0                             # m/s²
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s
    mount: tuple[float, float, float] = (0.0, 0.0, 0.0)      # Grad: Roll, Pitch, Yaw
    delay: float = 0.0                                   # s


@dataclass
class Randomization:
    """Alles, was für einen Durchlauf gewürfelt wird. Die Standardwerte sind das Modell ohne Zufall."""
    torso_mass: float = 1.0
    torso_com: tuple[float, float] = (0.0, 0.0)
    leg_mass: float = 1.0
    floor_friction: float = FLOOR_FRICTION
    floor_solref: float = NOMINAL_SOLREF
    servo: ServoParams = field(default_factory=ServoParams)
    period_jitter: float = 0.0
    pushes: list[Push] = field(default_factory=list)
    seed: int = 0  # für das Zittern des Steuertakts während des Durchlaufs
    imu: ImuErrors = field(default_factory=ImuErrors)


def sample(
    rng: np.random.Generator, pushes: bool = True, duration: float = 20.0, max_force: float = PUSH_FORCE[1],
) -> Randomization:
    """Neue Werte würfeln. duration (s): bis wann Stöße geplant werden, max_force (N): stärkster Stoß.

    Die Stöße werden zuletzt gewürfelt. Mit gleichem rng ist der Hund deshalb
    mit und ohne Stöße derselbe, nur die Stöße kommen dazu.
    """
    u = rng.uniform
    n = len(JOINT_ORDER)
    servo = ServoParams(
        stall_torque=u(*STALL_TORQUE),
        max_speed=math.radians(u(*MAX_SPEED)),
        kp=SERVO_KP * u(*GAIN),
        kd=SERVO_KD * u(*GAIN),
        deadband=np.radians(u(*DEADBAND, n)),
        dead_time=2 * int(rng.integers(DEAD_TIME[0] // 2 + 1, DEAD_TIME[1] // 2 + 1)) / 1000,
        offsets=np.radians(u(-CALIBRATION, CALIBRATION, n)),
    )
    r = Randomization(
        torso_mass=u(*TORSO_MASS),
        torso_com=(u(-TORSO_COM, TORSO_COM), u(-TORSO_COM, TORSO_COM)),
        leg_mass=u(*LEG_MASS),
        floor_friction=u(*FLOOR),
        floor_solref=u(*FLOOR_SOLREF),
        servo=servo,
        period_jitter=PERIOD_JITTER,
        seed=int(rng.integers(2**31)),
    )
    # Eigene Zufallszahlen aus seed: So bleiben alle anderen Werte (und die
    # Ausgangswerte aus Phase 1) gleich, und der Sensor ist mit und ohne Stöße derselbe.
    r.imu = _sample_imu(np.random.default_rng((r.seed, 1)))
    if pushes:
        r.pushes = sample_pushes(rng, duration, max_force)
    return r


def sample_pushes(rng: np.random.Generator, duration: float, max_force: float = PUSH_FORCE[1]) -> list[Push]:
    """Stöße alle 3 bis 6 s bis `duration` (s), waagrecht in zufälliger Richtung, 0 bis max_force N."""
    u = rng.uniform
    planned = []
    t = u(*PUSH_EVERY)
    while t < duration:
        angle = u(0.0, 2 * math.pi)
        force = u(PUSH_FORCE[0], max_force)
        planned.append(Push(t, (force * math.cos(angle), force * math.sin(angle), 0.0), u(*PUSH_DURATION)))
        t += u(*PUSH_EVERY)
    return planned


def _sample_imu(rng: np.random.Generator) -> ImuErrors:
    u = rng.uniform
    return ImuErrors(
        gyro_noise=u(*IMU_GYRO_NOISE),
        accel_noise=u(*IMU_ACCEL_NOISE),
        gyro_bias=tuple(u(-IMU_GYRO_BIAS, IMU_GYRO_BIAS, 3).tolist()),
        mount=tuple(u(-IMU_MOUNT, IMU_MOUNT, 3).tolist()),
        delay=2 * int(rng.integers(IMU_DELAY[0] // 2, IMU_DELAY[1] // 2 + 1)) / 1000,
    )


@lru_cache(maxsize=64)  # bleibt pro Durchlauf gleich, nicht jedes Mal neu rechnen
def _mount_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Drehmatrix Rumpf -> Sensor aus den Einbauwinkeln (Grad)."""
    r, p, y = np.radians((roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def imu_reading(
    sim: NovaSim, e: ImuErrors, rng: np.random.Generator,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Was der Lagesensor misst: (Drehrate rad/s, Beschleunigung m/s²) mit den Fehlern aus `e`."""
    gyro, accel = sim.imu(e.delay)
    if e == ImuErrors():  # perfekter Sensor, schneller Weg
        return gyro, accel
    turn = _mount_rotation(*e.mount).T  # Rumpf-Koordinaten -> Sensor-Koordinaten
    gyro = turn @ gyro + e.gyro_bias + rng.normal(0.0, e.gyro_noise, 3)
    accel = turn @ accel + rng.normal(0.0, e.accel_noise, 3)
    return tuple(gyro.tolist()), tuple(accel.tolist())


def apply(sim: NovaSim, r: Randomization) -> None:
    """Die Werte in eine frische Simulation schreiben (vor sim.reset() aufrufen)."""
    m = sim.model
    torso = m.body("torso").id
    m.body_mass[torso] *= r.torso_mass
    m.body_inertia[torso] *= r.torso_mass
    m.body_ipos[torso, :2] += np.array(r.torso_com) / 1000
    for leg in LEGS:
        for part in ("hip", "upper_leg", "lower_leg"):
            body = m.body(f"{leg}_{part}").id
            m.body_mass[body] *= r.leg_mass
            m.body_inertia[body] *= r.leg_mass
    floor = m.geom("floor").id
    m.geom_friction[floor, 0] = r.floor_friction
    m.geom_solref[floor] = (r.floor_solref, 1.0)
    mujoco.mj_setConst(m, sim.data)  # abgeleitete Größen (z. B. Gesamtmassen) neu rechnen
    sim.params = r.servo
