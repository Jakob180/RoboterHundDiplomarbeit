"""
Physik-Simulation von Nova in MuJoCo (nur am PC).

Spielt denselben Ablauf wie die Animation in gait.py ab (Stehen, Sitzen, Trab),
aber mit Schwerkraft, Bodenkontakt und Servos mit begrenzter Kraft und
Geschwindigkeit. So sieht man, ob Nova dabei stehen bleibt oder umfällt.

    python src/sim.py              # Fenster mit 3D-Ansicht, Ablauf wiederholt sich
    python src/sim.py --headless   # ohne Fenster, nur Auswertung
    python src/sim.py --headless --speed 100 --period 0.6   # andere Trab-Werte ausprobieren

Braucht `mujoco` im venv (pip install mujoco). Die Steuerung läuft wie später
am Pi mit 50 Hz: alle 20 ms neue Zielwinkel für die 12 Gelenke.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import NamedTuple

import mujoco

from gait import Frame, demo_frames
from kinematics import LegAngles
from nova_model import SERVO_SPEED, SERVO_TORQUE, build_mjcf
from servos import JOINT_ORDER

CONTROL_HZ = 50.0


class TorsoState(NamedTuple):
    """Lage des Rumpfs: Position in mm, Winkel in Grad wie bei BodyPose."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class NovaSim:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_mjcf())
        self.data = mujoco.MjData(self.model)
        self.torso_id = self.model.body("torso").id
        self._substeps = round(1.0 / (CONTROL_HZ * self.model.opt.timestep))
        self._qpos_adr = [self.model.joint(name).qposadr[0] for name in JOINT_ORDER]

    @staticmethod
    def joint_targets(angles: dict[str, LegAngles]) -> list[float]:
        """Kinematik-Winkel aller Beine -> 12 Werte in Radiant, Reihenfolge JOINT_ORDER."""
        return [math.radians(getattr(angles[name[:2]], name[3:])) for name in JOINT_ORDER]

    def reset(self, angles: dict[str, LegAngles], height: float) -> None:
        """Hund direkt in eine Pose stellen, Körpermitte auf `height` mm."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = height / 1000
        targets = self.joint_targets(angles)
        for adr, q in zip(self._qpos_adr, targets):
            self.data.qpos[adr] = q
        self.data.ctrl[:] = targets
        mujoco.mj_forward(self.model, self.data)

    def step(self, angles: dict[str, LegAngles]) -> None:
        """Neue Zielwinkel setzen und 20 ms Physik rechnen.

        Wie ein echter Servo fährt jedes Gelenk höchstens mit SERVO_SPEED auf sein Ziel zu.
        """
        max_change = SERVO_SPEED / CONTROL_HZ
        for i, target in enumerate(self.joint_targets(angles)):
            current = self.data.ctrl[i]
            self.data.ctrl[i] = current + max(-max_change, min(max_change, target - current))
        mujoco.mj_step(self.model, self.data, nstep=self._substeps)

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


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

def run_headless(frames: list[Frame]) -> None:
    sim = NovaSim()
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


def run_viewer(frames: list[Frame]) -> None:
    import mujoco.viewer

    sim = NovaSim()
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
    args = parser.parse_args()

    demo = demo_frames(CONTROL_HZ, args.speed, args.period, args.step_height, args.duty)
    if args.headless:
        run_headless(demo)
    else:
        run_viewer(demo)
