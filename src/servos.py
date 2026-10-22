"""
Servo-Schicht für Nova.

Alle Bewegungen (Gangmuster, inverse Kinematik, neuronales Netz) laufen über
`Servos.set_joint()` bzw. `Servos.set_all()`. Dort werden die Winkel IMMER auf
die Gelenkgrenzen begrenzt, bevor sie an die Hardware gehen.

Simulationsmodus (ohne Pi / ohne Servos): Umgebungsvariable NOVA_SIM=1 setzen,
z. B. am Pi oder unter Linux:
    NOVA_SIM=1 python3 src/servos.py

ACHTUNG: Noch nicht auf der echten Hardware getestet.
Die Grenzwerte unten sind Platzhalter und müssen für Nova ausgemessen werden,
am besten BEVOR die Beine montiert sind.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Joint:
    channel: int            # Kanal am PCA9685 (0-15)
    min_angle: float        # untere Grenze in Grad (Servo-Winkel)
    max_angle: float        # obere Grenze in Grad (Servo-Winkel)
    neutral: float          # Servo-Winkel in der Grundstellung
    inverted: bool = False  # True, wenn der Servo spiegelverkehrt eingebaut ist


# Beine: FL = vorne links, FR = vorne rechts, RL = hinten links, RR = hinten rechts
# Gelenke: shoulder = seitlich, upper = Oberschenkel, lower = Unterschenkel
# PLATZHALTER-WERTE: unbedingt am echten Hund ausmessen!
JOINTS: dict[str, Joint] = {
    "FL_shoulder": Joint(0,  60, 120, 90),
    "FL_upper":    Joint(1,  30, 150, 90),
    "FL_lower":    Joint(2,  30, 150, 90),

    "FR_shoulder": Joint(3,  60, 120, 90, inverted=True),
    "FR_upper":    Joint(4,  30, 150, 90, inverted=True),
    "FR_lower":    Joint(5,  30, 150, 90, inverted=True),

    "RL_shoulder": Joint(6,  60, 120, 90),
    "RL_upper":    Joint(7,  30, 150, 90),
    "RL_lower":    Joint(8,  30, 150, 90),

    "RR_shoulder": Joint(9,  60, 120, 90, inverted=True),
    "RR_upper":    Joint(10, 30, 150, 90, inverted=True),
    "RR_lower":    Joint(11, 30, 150, 90, inverted=True),
}

# Reihenfolge, in der z. B. das neuronale Netz seine 12 Ausgaben liefert
JOINT_ORDER: list[str] = list(JOINTS.keys())

# Pulsbreiten der Servos in Mikrosekunden (laut Datenblatt eurer Servos anpassen)
MIN_PULSE_US = 500
MAX_PULSE_US = 2500


# ---------------------------------------------------------------------------
# Hardware bzw. Simulation
# ---------------------------------------------------------------------------

SIM_MODE = os.environ.get("NOVA_SIM") == "1"


class _FakeServo:
    """Ersatz für einen echten Servo: merkt sich den Winkel und gibt ihn aus."""

    def __init__(self, channel: int) -> None:
        self.channel = channel
        self._angle: float | None = None

    @property
    def angle(self) -> float | None:
        return self._angle

    @angle.setter
    def angle(self, value: float | None) -> None:
        self._angle = value
        if value is None:
            print(f"[SIM] Kanal {self.channel:2d}: aus")
        else:
            print(f"[SIM] Kanal {self.channel:2d}: {value:6.1f}°")


def _create_backend():
    if SIM_MODE:
        return [_FakeServo(ch) for ch in range(16)]

    # Nur am Pi nötig. Läuft über I2C (PCA9685), nicht über RPi.GPIO/pigpio.
    from adafruit_servokit import ServoKit

    kit = ServoKit(channels=16)
    for joint in JOINTS.values():
        kit.servo[joint.channel].set_pulse_width_range(MIN_PULSE_US, MAX_PULSE_US)
    return kit.servo


# ---------------------------------------------------------------------------
# Öffentliche Schnittstelle
# ---------------------------------------------------------------------------

class Servos:
    def __init__(self) -> None:
        self._servo = _create_backend()

    @staticmethod
    def clamp(name: str, angle: float | None) -> float:
        """Begrenzt einen Winkel auf die Gelenkgrenzen. Ungültige Werte -> Grundstellung."""
        joint = JOINTS[name]
        if angle is None or not math.isfinite(angle):
            # z. B. NaN aus dem neuronalen Netz: lieber Grundstellung als Unsinn
            return joint.neutral
        return max(joint.min_angle, min(joint.max_angle, float(angle)))

    def set_joint(self, name: str, angle: float) -> float:
        """Setzt ein Gelenk. Gibt den tatsächlich gesetzten Winkel zurück."""
        joint = JOINTS[name]
        safe = self.clamp(name, angle)
        hw_angle = 180.0 - safe if joint.inverted else safe
        self._servo[joint.channel].angle = hw_angle
        return safe

    def set_all(self, angles: list[float]) -> list[float]:
        """Setzt alle 12 Gelenke in der Reihenfolge von JOINT_ORDER."""
        if len(angles) != len(JOINT_ORDER):
            raise ValueError(f"Erwarte {len(JOINT_ORDER)} Winkel, bekommen: {len(angles)}")
        return [self.set_joint(name, a) for name, a in zip(JOINT_ORDER, angles)]

    def neutral(self) -> None:
        """Fährt alle Gelenke in die Grundstellung."""
        for name, joint in JOINTS.items():
            self.set_joint(name, joint.neutral)

    def relax(self) -> None:
        """Schaltet das PWM-Signal ab, die Servos werden kraftlos."""
        for joint in JOINTS.values():
            self._servo[joint.channel].angle = None


if __name__ == "__main__":
    s = Servos()
    print("Grundstellung:")
    s.neutral()
    print("\nTest Begrenzung (200° und NaN):")
    print("FL_upper ->", s.set_joint("FL_upper", 200))
    print("FL_lower ->", s.set_joint("FL_lower", float("nan")))
    print("\nEntspannen:")
    s.relax()
