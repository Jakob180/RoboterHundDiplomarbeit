"""
MuJoCo-Modell von Nova, erzeugt aus den Maßen in kinematics.py.

Die Gelenke heißen wie in servos.py (FL_shoulder, ...) und haben denselben
Nullpunkt und dieselbe Drehrichtung wie die Kinematik-Winkel: 0/0/0 = Bein
senkrecht gestreckt. Die Winkel aus body_ik() gehen also direkt (in Radiant) an
die Simulation. Am echten Hund braucht es dafür noch die Kalibrierung.

Einfache Formen statt der echten STL-Teile. Massen und Servo-Werte sind
PLATZHALTER: mit den echten Teilen (Waage, Datenblatt) ersetzen.

Modell als Datei speichern, z. B. zum Anschauen in MuJoCos eigenem Viewer:
    python src/nova_model.py nova.xml
"""

from __future__ import annotations

import math

from kinematics import (
    BODY_LENGTH, BODY_WIDTH, HIP_LENGTH, LEGS, LOWER_LENGTH, SHOULDERS, UPPER_LENGTH,
)
from servos import JOINT_ORDER


# Massen in kg. PLATZHALTER, zusammen ca. 1,9 kg
BODY_MASS = 1.0    # Rumpf mit Pi, Akku, Elektronik und den 4 Schulterservos
HIP_MASS = 0.06    # Schulterstück
UPPER_MASS = 0.12  # Oberschenkel mit Hüft- und Knieservo
LOWER_MASS = 0.04  # Unterschenkel mit Fuß

# Servos. PLATZHALTER für einen 20-kg·cm-Servo bei 6 V, Datenblatt der echten Servos nehmen
SERVO_TORQUE = 2.0                # N·m, Haltemoment (20 kg·cm ≈ 2 N·m)
SERVO_SPEED = math.radians(375)   # rad/s, entspricht 0,16 s/60°
SERVO_KP = 30.0                   # N·m/rad, wie steif der Servo seine Position hält

FOOT_RADIUS = 10.0  # mm

# Gelenkbereiche als Kinematik-Winkel in Grad. PLATZHALTER: großzügig gewählt,
# die echten Grenzen kommen nach der Kalibrierung aus servos.py.
JOINT_RANGES = {"shoulder": (-45.0, 45.0), "upper": (-150.0, 90.0), "lower": (0.0, 160.0)}


def _m(mm: float) -> float:
    return mm / 1000.0


def _range(joint: str) -> str:
    lo, hi = JOINT_RANGES[joint]
    return f"{math.radians(lo):.4f} {math.radians(hi):.4f}"


def _leg_xml(leg: str) -> str:
    side = -1 if leg[1] == "R" else 1  # rechte Beine zeigen nach -y
    sx, sy, sz = (_m(v) for v in SHOULDERS[leg])
    hip, upper = _m(HIP_LENGTH), _m(UPPER_LENGTH)
    shin = _m(LOWER_LENGTH - FOOT_RADIUS)  # Kugelmitte, damit die Kugel unten am Fußpunkt aufliegt
    return f"""
      <body name="{leg}_hip" pos="{sx} {sy} {sz}">
        <joint name="{leg}_shoulder" axis="{side} 0 0" range="{_range('shoulder')}"/>
        <geom type="capsule" fromto="0 0 0 0 {side * hip} 0" size="0.012" mass="{HIP_MASS}"/>
        <body name="{leg}_upper_leg" pos="0 {side * hip} 0">
          <joint name="{leg}_upper" axis="0 -1 0" range="{_range('upper')}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-upper}" size="0.012" mass="{UPPER_MASS}"/>
          <body name="{leg}_lower_leg" pos="0 0 {-upper}">
            <joint name="{leg}_lower" axis="0 -1 0" range="{_range('lower')}"/>
            <!-- endet über der Fußkugel, damit nur der Fuß den Boden berührt -->
            <geom type="capsule" fromto="0 0 0 0 0 {-(shin - _m(FOOT_RADIUS))}" size="0.008" mass="{LOWER_MASS * 0.8}"/>
            <geom name="{leg}_foot" class="foot" type="sphere" pos="0 0 {-shin}"
                  size="{_m(FOOT_RADIUS)}" mass="{LOWER_MASS * 0.2}"/>
          </body>
        </body>
      </body>"""


def build_mjcf() -> str:
    """Das Modell als MJCF-Text (MuJoCos XML-Format)."""
    actuators = "\n".join(
        f'    <position name="{name}" joint="{name}" ctrlrange="{_range(name[3:])}"/>'
        for name in JOINT_ORDER
    )
    return f"""<mujoco model="nova">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .8 1" rgb2="1 1 1" width="256" height="256"/>
    <!-- Karomuster mit 10 cm Kästchen, damit man Wege abschätzen kann -->
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".7 .7 .7" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true"/>
  </asset>

  <default>
    <position kp="{SERVO_KP}" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>
    <default class="robot">
      <joint damping="0.1" armature="0.005"/>
      <!-- Roboterteile stoßen nur mit dem Boden zusammen, nicht miteinander -->
      <geom contype="1" conaffinity="0" rgba=".25 .45 .85 1"/>
      <default class="foot">
        <geom rgba=".15 .15 .15 1" friction="1 0.005 0.0001"/>
      </default>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="torso" pos="0 0 0.2" childclass="robot">
      <freejoint name="root"/>
      <geom type="box" size="{_m(BODY_LENGTH / 2 + 20)} {_m(BODY_WIDTH / 2)} 0.025"
            mass="{BODY_MASS}" rgba=".9 .9 .9 1"/>
      <!-- Licht und Kamera fahren mit dem Hund mit -->
      <light mode="trackcom" pos="0 0 2" dir="0 0 -1" directional="true"/>
      <camera name="side" mode="trackcom" pos="0 -0.65 0.08" xyaxes="1 0 0 0 0.12 1"/>
{''.join(_leg_xml(leg) for leg in LEGS)}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Aufruf: python src/nova_model.py <datei.xml>")
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write(build_mjcf())
    print(f"Gespeichert: {sys.argv[1]}")
