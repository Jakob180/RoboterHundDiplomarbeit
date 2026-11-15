"""
Schnittstelle zwischen Nova und dem neuronalen Netz (Plan Abschnitt 4.1, 4.2, 5.4).

Hier steht EINMAL, was das Netz sieht und wie aus seiner Ausgabe Gelenkwinkel
werden. Die Trainingsumgebung (model/nova_env.py) und später die Hauptschleife
am Pi benutzen genau diese Klassen. Sonst passiert der häufigste Fehler beim
Übertragen: eine andere Reihenfolge oder Skalierung am Pi als im Training.

Ablauf alle 20 ms:
    obs = mode.observe(oben, drehrate)   # oben aus imu.ComplementaryFilter, Drehrate rad/s
    aktion = netz(obs)                   # 12 Werte, gedacht -1..1
    winkel = mode.act(aktion, dt)        # Kinematik-Winkel je Bein, schon begrenzt

Zwei Stufen:
- ResidualMode (Stufe 1): Der klassische Trab läuft, das Netz verschiebt jeden
  Fuß ein Stück (±20 mm vor/zurück, ±10 mm seitlich, ±15 mm hoch/runter).
  Ausgabe 0 = genau der klassische Trab.
- FullMode (Stufe 2): Das Netz gibt die 12 Gelenkwinkel selbst vor (relativ zur
  Stehpose). Der Trab läuft nur als Taktgeber und als Vorbild mit.

Sicherheit: Ungültige Ausgaben (NaN) zählen als 0, alles wird auf -1..1
begrenzt, die Gelenkwinkel auf die Grenzen aus servos.py. Am echten Hund
begrenzt servos.py danach noch einmal. Noch nicht auf der Hardware getestet.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from gait import STAND, Trot
from kinematics import LEGS, LegAngles, body_ik, foot_ik
from servos import JOINT_ORDER, kinematic_limits

# Skalierung der Beobachtung (Plan 4.1). Fest statt gelernt, damit am Pi nichts fehlen kann.
GYRO_SCALE = 4.0                  # rad/s
MAX_SPEED = 200.0                 # mm/s, größtes Tempo im Befehl
MAX_YAW_RATE = math.degrees(0.5)  # °/s, größte Drehrate im Befehl (0,5 rad/s)
ANGLE_SCALE = 45.0                # Grad
OBS_CLIP = 5.0                    # größter Betrag eines Eingangs

# Befehl weich anfahren, damit der Trab nicht ruckt
SPEED_RAMP = 200.0                # mm/s pro s
YAW_RAMP = 60.0                   # °/s pro s

# Stufe 1: Korrektur je Fuß bei Ausgabe ±1, in mm (x vorne, y links, z oben)
FOOT_SCALE = (20.0, 10.0, 15.0)
SMOOTHING = 0.2                   # Anteil der vorigen Korrektur (leicht geglättet)

# Stufe 2: Gelenkwinkel bei Ausgabe ±1, in Grad relativ zur Stehpose
JOINT_SCALE = {"shoulder": 15.0, "upper": 30.0, "lower": 30.0}
MAX_JOINT_SPEED = 400.0           # °/s, höchstens 8° pro 20 ms (so schnell sind die Servos)

N_ACTIONS = len(JOINT_ORDER)
_LOW = np.array([kinematic_limits(name)[0] for name in JOINT_ORDER])
_HIGH = np.array([kinematic_limits(name)[1] for name in JOINT_ORDER])
_JOINT_SCALES = np.array([JOINT_SCALE[name[3:]] for name in JOINT_ORDER])


def joint_array(angles: dict[str, LegAngles]) -> np.ndarray:
    """{"FL": LegAngles, ...} -> 12 Winkel in Grad, Reihenfolge JOINT_ORDER."""
    return np.array([getattr(angles[name[:2]], name[3:]) for name in JOINT_ORDER], dtype=float)


def joint_dict(array: np.ndarray) -> dict[str, LegAngles]:
    """12 Winkel in Grad (Reihenfolge JOINT_ORDER) -> {"FL": LegAngles, ...}."""
    return {leg: LegAngles(*(float(a) for a in array[3 * i:3 * i + 3])) for i, leg in enumerate(LEGS)}


def clamp_angles(array: np.ndarray) -> np.ndarray:
    """Auf die Gelenkgrenzen aus servos.py begrenzen."""
    return np.clip(array, _LOW, _HIGH)


STAND_ANGLES = joint_array(body_ik(STAND.feet, STAND.body))


class _Mode:
    """Was beide Stufen gemeinsam haben: Befehl, Trab als Taktgeber, letzte Aktion."""

    n_obs = 34
    n_actions = N_ACTIONS

    def __init__(self, period: float = 0.5, step_height: float = 25.0, duty: float = 0.6) -> None:
        self._trot_settings = (period, step_height, duty)
        self.reset()

    def reset(self) -> None:
        self.trot = Trot(STAND, *self._trot_settings)
        self.speed = 0.0                          # mm/s, aktueller (weich angefahrener) Befehl
        self.yaw_rate = 0.0                       # °/s
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.last_action = np.zeros(N_ACTIONS)    # letzte Ausgabe des Netzes, begrenzt
        self.angles = STAND_ANGLES.copy()         # zuletzt befohlene Gelenkwinkel (Grad)

    def set_command(self, speed: float, yaw_rate: float) -> None:
        """Neuer Befehl: Tempo in mm/s (+ = vorwärts), Drehrate in °/s (+ = links)."""
        self.target_speed = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self.target_yaw_rate = float(np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))

    def _ramp(self, dt: float) -> None:
        step = SPEED_RAMP * dt
        self.speed += float(np.clip(self.target_speed - self.speed, -step, step))
        step = YAW_RAMP * dt
        self.yaw_rate += float(np.clip(self.target_yaw_rate - self.yaw_rate, -step, step))

    @staticmethod
    def _clean(action) -> np.ndarray:
        """Ausgabe des Netzes prüfen: NaN/unendlich -> alles 0, sonst auf -1..1 begrenzen."""
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (N_ACTIONS,):
            raise ValueError(f"Erwarte {N_ACTIONS} Werte vom Netz, bekommen: {a.size}")
        if not np.all(np.isfinite(a)):
            return np.zeros(N_ACTIONS)
        return np.clip(a, -1.0, 1.0)

    def _base_observation(self, up, gyro) -> list[float]:
        """Die ersten 22 Eingänge, gleich in beiden Stufen."""
        phase = 2 * math.pi * self.trot.phase
        return [
            *up,                                          # 0-2:   Richtung "oben" im Rumpf (Filter)
            *(g / GYRO_SCALE for g in gyro),              # 3-5:   Drehrate Roll, Pitch, Yaw
            self.speed / MAX_SPEED,                       # 6:     Befehl Tempo
            self.yaw_rate / MAX_YAW_RATE,                 # 7:     Befehl Drehrate
            math.sin(phase), math.cos(phase),             # 8-9:   Phase des Trabs
            *self.last_action,                            # 10-21: letzte Ausgabe des Netzes
        ]

    @staticmethod
    def _finish(values: list[float]) -> np.ndarray:
        return np.clip(np.array(values, dtype=np.float32), -OBS_CLIP, OBS_CLIP)


class ResidualMode(_Mode):
    """Stufe 1: klassischer Trab, das Netz korrigiert die Fußpunkte."""

    def reset(self) -> None:
        super().reset()
        self.correction = np.zeros(N_ACTIONS)  # mm je Fuß (x, y, z), geglättet

    def observe(self, up, gyro) -> np.ndarray:
        """34 Eingänge: Grundwerte + zuletzt befohlene Gelenkwinkel relativ zur Stehpose."""
        values = self._base_observation(up, gyro)
        values += list((self.angles - STAND_ANGLES) / ANGLE_SCALE)   # 22-33
        return self._finish(values)

    def act(self, action, dt: float) -> dict[str, LegAngles]:
        """Ausgabe des Netzes -> Gelenkwinkel. Trab läuft dabei dt Sekunden weiter."""
        a = self._clean(action)
        self._ramp(dt)
        pose = self.trot.update(dt, self.speed, self.yaw_rate)
        self.correction = (1 - SMOOTHING) * a * np.tile(FOOT_SCALE, len(LEGS)) + SMOOTHING * self.correction

        angles = {}
        for i, leg in enumerate(LEGS):
            fx, fy, fz = pose.feet[leg]
            dx, dy, dz = self.correction[3 * i:3 * i + 3]
            try:
                angles[leg] = foot_ik(leg, (fx + dx, fy + dy, fz + dz), pose.body)
            except ValueError:
                # Fuß mit Korrektur nicht erreichbar: dieses Bein ohne Korrektur
                self.correction[3 * i:3 * i + 3] = 0.0
                try:
                    angles[leg] = foot_ik(leg, pose.feet[leg], pose.body)
                except ValueError:
                    angles[leg] = LegAngles(*self.angles[3 * i:3 * i + 3])  # notfalls stehen bleiben

        self.angles = clamp_angles(joint_array(angles))
        self.last_action = a
        return joint_dict(self.angles)


class FullMode(_Mode):
    """Stufe 2: Das Netz gibt die Gelenkwinkel vor, der Trab ist Taktgeber und Vorbild."""

    def reset(self) -> None:
        super().reset()
        self.reference = STAND_ANGLES.copy()  # Gelenkwinkel des Trabs (Vorbild fürs Nachahmen)
        self._history: deque[list[float]] = deque(maxlen=2)  # Lagesensor der letzten 2 Schritte

    def observe(self, up, gyro) -> np.ndarray:
        """34 Eingänge: Grundwerte + Lagesensor von vor 1 und vor 2 Schritten."""
        now = [*up, *(g / GYRO_SCALE for g in gyro)]
        history = list(self._history) or [now]
        values = self._base_observation(up, gyro)
        values += history[0] + history[-1]   # 22-27: vor 1 Schritt, 28-33: vor 2 Schritten
        self._history.appendleft(now)
        return self._finish(values)

    def act(self, action, dt: float) -> dict[str, LegAngles]:
        """Ausgabe des Netzes -> Gelenkwinkel, höchstens MAX_JOINT_SPEED schnell."""
        a = self._clean(action)
        self._ramp(dt)
        pose = self.trot.update(dt, self.speed, self.yaw_rate)
        self.reference = clamp_angles(joint_array(body_ik(pose.feet, pose.body)))

        target = clamp_angles(STAND_ANGLES + a * _JOINT_SCALES)
        step = MAX_JOINT_SPEED * dt
        self.angles = self.angles + np.clip(target - self.angles, -step, step)
        self.last_action = a
        return joint_dict(self.angles)
