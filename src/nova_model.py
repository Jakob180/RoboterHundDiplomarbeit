"""
MuJoCo-Modell von Nova, erzeugt aus den Maßen in kinematics.py.

Die Gelenke heißen wie in servos.py (FL_shoulder, ...) und haben denselben
Nullpunkt und dieselbe Drehrichtung wie die Kinematik-Winkel: 0/0/0 = Bein
senkrecht gestreckt. Die Winkel aus body_ik() gehen also direkt (in Radiant) an
die Simulation. Die Gelenkgrenzen kommen aus der Kalibrierung in servos.py.

Die Antriebe sind reine Drehmoment-Motoren: Wie stark jeder Servo drückt,
rechnet das Servo-Modell in sim.py (Totzeit, Kennlinie, Totzone).

Sensoren wie am echten Hund: Lagesensor (Gyroskop + Beschleunigung) im Rumpf
und ein Abstandssensor vorne.

Einfache Formen statt der echten STL-Teile. Massen und Lage von Akku und
Powerbank sind PLATZHALTER: mit den echten Teilen (Waage) ersetzen.

Modell als Datei speichern, z. B. zum Anschauen in MuJoCos eigenem Viewer:
    python src/nova_model.py nova.xml
"""

from __future__ import annotations

import math

from kinematics import (
    BODY_LENGTH, BODY_WIDTH, HIP_LENGTH, LEGS, LOWER_LENGTH, SHOULDERS, UPPER_LENGTH,
)
from servos import JOINT_ORDER, kinematic_limits


# Massen in kg. PLATZHALTER (geschätzt), zusammen ca. 2,6 kg
FRAME_MASS = 1.03      # Rumpf mit Pi, Elektronik, Kabeln und den 4 Schulterservos
BATTERY_MASS = 0.25    # 2S-LiPo für die Servos
POWERBANK_MASS = 0.22  # Powerbank für den Pi
HIP_MASS = 0.06        # Schulterstück
UPPER_MASS = 0.17      # Oberschenkel mit Hüft- und Knieservo
LOWER_MASS = 0.045     # Unterschenkel mit Fuß

# Lage im Rumpf in mm (x vorne, y links, z oben, ab der Rumpfmitte). PLATZHALTER
BATTERY_POS = (-20.0, 0.0, -12.0)    # unten, etwas hinten
POWERBANK_POS = (40.0, 0.0, 12.0)    # oben, etwas vorne
IMU_POS = (0.0, 0.0, 20.0)           # GY-521 in der Mitte auf dem Rahmen
RANGE_POS = (BODY_LENGTH / 2 + 20, 0.0, 0.0)  # vorne an der Nase, schaut nach vorne

# Servos: Miuzei 25 kg laut Datenblatt, umgerechnet auf 7,4 V (2S-LiPo, halb voll)
SERVO_TORQUE = 2.59               # N·m, Blockiermoment (4,8 V: 2,30, 8,4 V: 2,70)
SERVO_SPEED = math.radians(443)   # rad/s ohne Last (4,8 V: 400 °/s, 8,4 V: 462 °/s)
SERVO_KP = 30.0                   # N·m/rad, wie fest der Servo seine Position hält. PLATZHALTER
# Bremswirkung des Motors: Bei voller Drehzahl frisst sie das ganze Moment auf.
# Damit folgt die Kennlinie (Moment fällt mit der Drehzahl) direkt aus dem Regler.
SERVO_KD = SERVO_TORQUE / SERVO_SPEED  # N·m·s/rad
SERVO_DEADBAND = math.radians(0.27)    # rad, Totzone 3 µs von 2000 µs für 180°
SERVO_DEAD_TIME = 0.020                # s vom Befehl bis zur Reaktion. PLATZHALTER, messen

# Boden: Laminat ungefähr. In der Randomisierung 0,4 bis 1,2.
FLOOR_FRICTION = 0.8

FOOT_RADIUS = 10.0  # mm


def _m(mm: float) -> float:
    return mm / 1000.0


def _mm(point: tuple[float, float, float]) -> str:
    return " ".join(str(_m(v)) for v in point)


def _range(joint: str) -> str:
    """Gelenkgrenzen aus servos.py in Radiant, z. B. für FL_upper."""
    lo, hi = kinematic_limits(joint)
    return f"{math.radians(lo):.6f} {math.radians(hi):.6f}"


def _leg_xml(leg: str) -> str:
    side = -1 if leg[1] == "R" else 1  # rechte Beine zeigen nach -y
    sx, sy, sz = (_m(v) for v in SHOULDERS[leg])
    hip, upper = _m(HIP_LENGTH), _m(UPPER_LENGTH)
    shin = _m(LOWER_LENGTH - FOOT_RADIUS)  # Kugelmitte, damit die Kugel unten am Fußpunkt aufliegt
    return f"""
      <body name="{leg}_hip" pos="{sx} {sy} {sz}">
        <joint name="{leg}_shoulder" axis="{side} 0 0" range="{_range(leg + '_shoulder')}"/>
        <geom type="capsule" fromto="0 0 0 0 {side * hip} 0" size="0.012" mass="{HIP_MASS}"/>
        <body name="{leg}_upper_leg" pos="0 {side * hip} 0">
          <joint name="{leg}_upper" axis="0 -1 0" range="{_range(leg + '_upper')}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-upper}" size="0.012" mass="{UPPER_MASS}"/>
          <body name="{leg}_lower_leg" pos="0 0 {-upper}">
            <joint name="{leg}_lower" axis="0 -1 0" range="{_range(leg + '_lower')}"/>
            <!-- endet über der Fußkugel, damit nur der Fuß den Boden berührt -->
            <geom type="capsule" fromto="0 0 0 0 0 {-(shin - _m(FOOT_RADIUS))}" size="0.008" mass="{LOWER_MASS * 0.8}"/>
            <geom name="{leg}_foot" class="foot" type="sphere" pos="0 0 {-shin}"
                  size="{_m(FOOT_RADIUS)}" mass="{LOWER_MASS * 0.2}"/>
          </body>
        </body>
      </body>"""


def build_mjcf(self_collision: bool = False) -> str:
    """Das Modell als MJCF-Text (MuJoCos XML-Format).

    self_collision=True: Die Beine können auch aneinander und am Rumpf anstoßen
    (langsamer). Sonst stoßen die Roboterteile nur mit dem Boden zusammen.
    """
    # Moment in N·m, gerechnet und begrenzt wird es im Servo-Modell in sim.py
    actuators = "\n".join(f'    <motor name="{name}" joint="{name}"/>' for name in JOINT_ORDER)
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
    <default class="robot">
      <!-- armature: Trägheit von Motor und Getriebe. damping klein, die Bremswirkung
           des Motors steckt im Servo-Modell (SERVO_KD) -->
      <joint damping="0.01" armature="0.005"/>
      <geom contype="1" conaffinity="{1 if self_collision else 0}" rgba=".25 .45 .85 1"/>
      <default class="foot">
        <geom rgba=".15 .15 .15 1"/>
      </default>
      <default class="cargo">
        <!-- liegt im Rumpf, zählt nur als Masse -->
        <geom type="box" contype="0" conaffinity="0" group="3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <!-- priority: Reibung und Weichheit des Bodens gelten für den Kontakt, nicht die der Füße -->
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" priority="1"
          friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <body name="torso" pos="0 0 0.2" childclass="robot">
      <freejoint name="root"/>
      <geom name="frame" type="box" size="{_m(BODY_LENGTH / 2 + 20)} {_m(BODY_WIDTH / 2)} 0.025"
            mass="{FRAME_MASS}" rgba=".9 .9 .9 1"/>
      <geom name="battery" class="cargo" pos="{_mm(BATTERY_POS)}" size="0.069 0.023 0.0125"
            mass="{BATTERY_MASS}" rgba=".8 .2 .2 1"/>
      <geom name="powerbank" class="cargo" pos="{_mm(POWERBANK_POS)}" size="0.045 0.03 0.01"
            mass="{POWERBANK_MASS}" rgba=".2 .2 .2 1"/>
      <site name="imu" pos="{_mm(IMU_POS)}" size="0.008" rgba="0 .8 0 1"/>
      <site name="range" pos="{_mm(RANGE_POS)}" zaxis="1 0 0" size="0.006" rgba="1 .5 0 1"/>
      <!-- Licht und Kamera fahren mit dem Hund mit -->
      <light mode="trackcom" pos="0 0 2" dir="0 0 -1" directional="true"/>
      <camera name="side" mode="trackcom" pos="0 -0.65 0.08" xyaxes="1 0 0 0 0.12 1"/>
{''.join(_leg_xml(leg) for leg in LEGS)}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu"/>                <!-- rad/s -->
    <accelerometer name="accel" site="imu"/>      <!-- m/s², in Ruhe +9,81 in z -->
    <rangefinder name="range" site="range"/>      <!-- m, -1 = nichts getroffen -->
  </sensor>
</mujoco>
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Aufruf: python src/nova_model.py <datei.xml>")
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write(build_mjcf())
    print(f"Gespeichert: {sys.argv[1]}")
