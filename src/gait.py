"""
Grundposen (Stehen, Sitzen) und ein einfacher Trab für Nova.

Alle Fußpunkte sind Boden-Koordinaten in mm: Ursprung am Boden senkrecht unter
der Körpermitte, x vorne, y links, z oben. `kinematics.body_ik()` macht daraus
die Gelenkwinkel der 4 Beine.

Animation am PC (braucht matplotlib im venv, siehe CLAUDE.md):
    python src/gait.py                  # Fenster mit Animation
    python src/gait.py --gif nova.gif   # stattdessen als GIF speichern

Reine Kinematik ohne Physik: Ob Nova dabei das Gleichgewicht hält, zeigt erst
die Simulation (PyBullet) bzw. der echte Hund. Die Ausgaben sind
Kinematik-Winkel, noch keine Servo-Winkel. Nicht auf der Hardware getestet.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from kinematics import (
    BODY_LENGTH, BODY_WIDTH, HIP_LENGTH, LEGS, BodyPose, LegAngles, Vec3, body_ik, body_points,
)


# ---------------------------------------------------------------------------
# Grundposen
# ---------------------------------------------------------------------------

class Pose(NamedTuple):
    feet: dict[str, Vec3]  # Fußpunkte in Boden-Koordinaten
    body: BodyPose


def _feet(front_x: float = 0.0, rear_x: float = 0.0) -> dict[str, Vec3]:
    """Füße senkrecht unter den Hüftgelenken, vorne/hinten in x verschiebbar."""
    y = BODY_WIDTH / 2 + HIP_LENGTH
    return {
        "FL": (BODY_LENGTH / 2 + front_x, y, 0.0),
        "FR": (BODY_LENGTH / 2 + front_x, -y, 0.0),
        "RL": (-BODY_LENGTH / 2 + rear_x, y, 0.0),
        "RR": (-BODY_LENGTH / 2 + rear_x, -y, 0.0),
    }


# PLATZHALTER: Höhen und Winkel in der Simulation bzw. am echten Hund anpassen.
STAND = Pose(_feet(), BodyPose(height=170.0))
SIT = Pose(_feet(front_x=10.0, rear_x=30.0), BodyPose(height=125.0, pitch=-25.0))  # Nase hoch


def blend(a: Pose, b: Pose, s: float) -> Pose:
    """Zwischenpose von a (s=0) nach b (s=1). Fährt weich an und bremst weich ab."""
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)

    def mix(p: float, q: float) -> float:
        return p + (q - p) * s

    feet = {leg: tuple(mix(p, q) for p, q in zip(a.feet[leg], b.feet[leg])) for leg in LEGS}
    return Pose(feet, BodyPose(*(mix(p, q) for p, q in zip(a.body, b.body))))


# ---------------------------------------------------------------------------
# Trab
# ---------------------------------------------------------------------------

# Diagonale Beinpaare sind um eine halbe Periode versetzt: erst FL+RR, dann FR+RL
_PHASE_OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}


class Trot:
    """Einfacher Trab. `update()` regelmäßig aufrufen, z. B. alle 20 ms.

    Jeder Fuß steht den Anteil `duty` einer Periode am Boden (Standphase) und
    schwingt den Rest nach vorne (Schwungphase). Bei duty > 0,5 stehen beim
    Wechsel der Paare kurz alle 4 Füße am Boden, das macht den Trab ruhiger.
    In der Standphase bleibt der Fuß an seiner Stelle am Boden und der Körper
    bewegt sich darüber hinweg. Deshalb merkt sich die Klasse die Füße in
    Weltkoordinaten (x ab dem Startpunkt) und den Weg `distance`, den die
    Körpermitte zurückgelegt hat.
    """

    def __init__(
        self, base: Pose = STAND, period: float = 0.5, step_height: float = 25.0, duty: float = 0.6,
    ) -> None:
        self.base = base                # Pose, um die herum getrabt wird
        self.period = period            # s für einen ganzen Zyklus (jedes Bein 1 Schritt)
        self.step_height = step_height  # mm, wie hoch die Füße gehoben werden
        self.duty = duty                # Anteil der Periode, in dem ein Fuß am Boden steht
        self.distance = 0.0             # mm, Weg der Körpermitte seit dem Start
        self._phase = 0.0               # 0..1 innerhalb einer Periode
        self._foot_x = {leg: base.feet[leg][0] for leg in LEGS}  # Welt-x der Füße
        self._liftoff_x: dict[str, float] = {}  # Welt-x beim Abheben, nur in der Schwungphase
        self._target_x: dict[str, float] = {}

    def update(self, dt: float, speed: float) -> Pose:
        """Einen Zeitschritt dt (s) weiter mit Geschwindigkeit speed (mm/s, + = vorwärts)."""
        self.distance += speed * dt
        self._phase = (self._phase + dt / self.period) % 1.0

        feet = {}
        for leg in LEGS:
            home_x, home_y, home_z = self.base.feet[leg]
            p = (self._phase + _PHASE_OFFSET[leg]) % 1.0
            z = home_z

            if p >= self.duty:  # Schwungphase
                if leg not in self._liftoff_x:
                    self._liftoff_x[leg] = self._foot_x[leg]
                # Ziel: beim Aufsetzen eine halbe Schrittlänge vor der Grundposition
                # stehen, dann ist der Fuß beim Abheben gleich weit dahinter.
                time_left = (1.0 - p) * self.period
                stride = speed * self.period * self.duty  # Weg des Körpers während einer Standphase
                self._target_x[leg] = self.distance + speed * time_left + home_x + stride / 2

                # Zykloide: Der Fuß hebt ohne Ruck ab und setzt ohne Rutschen auf
                u = (p - self.duty) / (1.0 - self.duty)
                progress = u - math.sin(2 * math.pi * u) / (2 * math.pi)
                start = self._liftoff_x[leg]
                self._foot_x[leg] = start + (self._target_x[leg] - start) * progress
                z += self.step_height * (1 - math.cos(2 * math.pi * u)) / 2

            elif leg in self._liftoff_x:  # gerade aufgesetzt
                del self._liftoff_x[leg]
                self._foot_x[leg] = self._target_x[leg]

            feet[leg] = (self._foot_x[leg] - self.distance, home_y, z)

        return Pose(feet, self.base.body)


# ---------------------------------------------------------------------------
# Animation (nur am PC)
# ---------------------------------------------------------------------------

class Frame(NamedTuple):
    time: float
    text: str
    pose: Pose
    angles: dict[str, LegAngles]
    distance: float  # mm, wie weit der Körper schon gelaufen ist


def demo_frames(
    fps: float = 50.0, speed: float = 150.0, period: float = 0.5, step_height: float = 25.0,
    duty: float = 0.6,
) -> list[Frame]:
    """Ablauf: Stehen, Hinsetzen, Aufstehen, Trab (anlaufen, laufen, anhalten), Stehen.

    speed (mm/s), period (s), step_height (mm) und duty gelten für den Trab.
    """
    dt = 1.0 / fps
    frames: list[Frame] = []

    def add(text: str, pose: Pose, distance: float = 0.0) -> None:
        frames.append(Frame(len(frames) * dt, text, pose, body_ik(pose.feet, pose.body), distance))

    def move(a: Pose, b: Pose, seconds: float, text: str, distance: float = 0.0) -> None:
        n = round(seconds * fps)
        for i in range(n):
            add(text, blend(a, b, (i + 1) / n), distance)

    move(STAND, STAND, 1.0, "Stehen")
    move(STAND, SIT, 2.0, "Hinsetzen")
    move(SIT, SIT, 1.5, "Sitzen")
    move(SIT, STAND, 2.0, "Aufstehen")
    move(STAND, STAND, 0.5, "Stehen")

    trot = Trot(STAND, period, step_height, duty)
    for i in range(round(9.0 * fps)):
        t = i * dt
        if t < 2.0:
            v, text = speed * t / 2.0, "Trab: anlaufen"
        elif t < 6.0:
            v, text = speed, "Trab"
        elif t < 8.0:
            v, text = speed * (8.0 - t) / 2.0, "Trab: abbremsen"
        else:
            # Auf der Stelle weitertrappeln, bis alle Füße wieder in der
            # Grundposition stehen, und dabei die Schritthöhe auf 0 bringen
            v, text = 0.0, "Trab: anhalten"
            trot.step_height = step_height * (9.0 - t - dt)
        add(f"{text} ({v:3.0f} mm/s)", trot.update(dt, v), trot.distance)

    move(STAND, STAND, 1.0, "Stehen", trot.distance)
    return frames


def animate(frames: list[Frame], fps: float, gif: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes((0.0, 0.0, 0.62, 0.95), projection="3d")
    ax_angles = fig.add_axes((0.68, 0.3, 0.3, 0.6))
    title = fig.suptitle("")

    # 3D-Ansicht: Kamera fährt mit dem Körper mit
    ax.set_ylim(-200, 200)
    ax.set_zlim(0, 300)
    ax.set_box_aspect((500, 400, 300))  # gleicher Maßstab auf allen Achsen
    ax.view_init(elev=15, azim=-75)     # schräg von rechts vorne
    ax.set_xlabel("x vorne [mm]")
    ax.set_ylabel("y links [mm]")
    ax.set_zlabel("z [mm]")
    # Querstriche am Boden, damit man sieht, dass der Hund vorankommt
    end = max(f.distance for f in frames) + 300
    for x in range(-300, int(end), 100):
        ax.plot([x, x], [-200, 200], [0, 0], color="lightgray", lw=1)

    body_line, = ax.plot([], [], [], color="black", lw=3)
    leg_lines = {
        leg: ax.plot([], [], [], "-o", lw=3, ms=4, color="tab:orange" if leg[1] == "R" else "tab:blue")[0]
        for leg in LEGS
    }

    # Gelenkwinkel über die Zeit: FL durchgezogen, RL gestrichelt
    times = [f.time for f in frames]
    for leg, style in (("FL", "-"), ("RL", "--")):
        for joint, color in zip(LegAngles._fields, ("tab:green", "tab:blue", "tab:red")):
            values = [getattr(f.angles[leg], joint) for f in frames]
            ax_angles.plot(times, values, style, color=color, label=f"{leg} {joint}")
    cursor = ax_angles.axvline(0.0, color="black", lw=1)
    ax_angles.set_xlabel("Zeit [s]")
    ax_angles.set_ylabel("Kinematik-Winkel [°]")
    ax_angles.grid(True, alpha=0.3)
    ax_angles.legend(fontsize="small", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    def update(i: int):
        frame = frames[i]
        points = body_points(frame.angles, frame.pose.body)
        shift = frame.distance  # Boden-Koordinaten -> Welt
        for leg, line in leg_lines.items():
            xs, ys, zs = zip(*points[leg])
            line.set_data_3d([x + shift for x in xs], ys, zs)
        outline = [points[leg][0] for leg in ("FL", "FR", "RR", "RL", "FL")]
        xs, ys, zs = zip(*outline)
        body_line.set_data_3d([x + shift for x in xs], ys, zs)
        ax.set_xlim(shift - 250, shift + 250)
        title.set_text(f"{frame.text}   t = {frame.time:4.1f} s")
        cursor.set_xdata([frame.time])
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    if gif:
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=70)
        print(f"Gespeichert: {gif}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animation: Stehen, Sitzen, Trab")
    parser.add_argument("--gif", help="Animation als GIF speichern statt anzeigen")
    args = parser.parse_args()

    fps = 25.0 if args.gif else 50.0
    animate(demo_frames(fps), fps, args.gif)
