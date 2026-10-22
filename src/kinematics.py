"""
Inverse Kinematik für Nova: ein Bein und der ganze Körper.

`leg_ik()` rechnet eine Fußposition in die 3 Gelenkwinkel um, `leg_fk()` macht
das Umgekehrte. `body_ik()` macht das für alle 4 Beine auf einmal, wobei der
Körper geneigt sein darf. Visualisierung eines Beins am PC (braucht matplotlib,
z. B. im venv):
    python src/kinematics.py

Koordinaten eines Beins in mm, Ursprung auf der Drehachse der Schulter:
    x: nach vorne
    y: seitlich nach AUSSEN (linke Beine nach links, rechte Beine nach rechts)
    z: nach oben, der Fuß liegt also bei negativem z
Weil y immer nach außen zeigt, rechnen alle vier Beine gleich.

Gelenkwinkel in Grad. Das sind Kinematik-Winkel, KEINE Servo-Winkel:
    shoulder: Drehung um die x-Achse. 0 = Bein hängt senkrecht, + = Fuß nach außen
    upper:    Oberschenkel gegenüber senkrecht nach unten, + = nach vorne
    lower:    Knie, Winkel zwischen Ober- und Unterschenkel. 0 = gestreckt,
              + = Knie zeigt nach hinten (wie beim Spot Micro)
Die Umrechnung in Servo-Winkel (Nullpunkt, Drehrichtung, Spiegelung rechts)
hängt davon ab, wie die Servos montiert sind, und kommt mit der Kalibrierung.
Die Gelenkgrenzen werden hier NICHT geprüft, das macht servos.py.
"""

from __future__ import annotations

import math
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Maße
# ---------------------------------------------------------------------------

# Beinlängen in mm. PLATZHALTER: Spot-Micro-Standardmaße aus
# mike4192/spot_micro_kinematics_python. Für Nova aus den STL-Dateien abmessen!
HIP_LENGTH = 55.0     # seitlicher Versatz: Schulterachse -> Hüftgelenk
UPPER_LENGTH = 107.5  # Oberschenkel: Hüftgelenk -> Knie
LOWER_LENGTH = 130.0  # Unterschenkel: Knie -> Fuß

# Körpermaße in mm: Abstand der Schulterachsen. PLATZHALTER wie oben.
BODY_LENGTH = 186.0   # vorne -> hinten
BODY_WIDTH = 78.0     # links -> rechts

# Rundungsfehler, die bei voll gestrecktem Bein noch als erreichbar gelten
_EPS = 1e-9

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Kinematik
# ---------------------------------------------------------------------------

class LegAngles(NamedTuple):
    """Gelenkwinkel eines Beins in Grad (Kinematik-Winkel, keine Servo-Winkel)."""
    shoulder: float
    upper: float
    lower: float


def leg_ik(x: float, y: float, z: float) -> LegAngles:
    """Fußposition (mm, Bein-Koordinaten) -> Gelenkwinkel (Grad).

    Wirft ValueError, wenn der Fuß dort nicht hinkommt.
    """
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(f"Ungültige Fußposition: ({x}, {y}, {z})")

    # 1) Schulter: In der y-z-Ebene liegt der Fuß um HIP_LENGTH seitlich versetzt.
    #    Was übrig bleibt, ist die Länge des Beins nach unten in seiner Ebene.
    dist_yz_sq = y * y + z * z
    if dist_yz_sq < HIP_LENGTH ** 2:
        raise ValueError(
            f"Fuß zu nah an der Schulterachse: {math.sqrt(dist_yz_sq):.1f} mm, "
            f"mindestens {HIP_LENGTH:.1f} mm"
        )
    down = math.sqrt(dist_yz_sq - HIP_LENGTH ** 2)
    shoulder = math.atan2(z, y) - math.atan2(-down, HIP_LENGTH)
    shoulder = math.remainder(shoulder, 2 * math.pi)  # auf -180..180° bringen

    # 2) Oberschenkel und Knie: ebenes Zwei-Gelenk-Problem in der Beinebene
    #    (nach vorne = x, nach unten = down), Knie über den Kosinussatz.
    dist_sq = x * x + down * down
    cos_knee = (dist_sq - UPPER_LENGTH ** 2 - LOWER_LENGTH ** 2) / (2 * UPPER_LENGTH * LOWER_LENGTH)
    if abs(cos_knee) > 1.0 + _EPS:
        if cos_knee > 0:
            limit = f"höchstens {UPPER_LENGTH + LOWER_LENGTH:.1f} mm"
        else:
            limit = f"mindestens {abs(UPPER_LENGTH - LOWER_LENGTH):.1f} mm"
        raise ValueError(f"Fuß nicht erreichbar: Hüfte -> Fuß {math.sqrt(dist_sq):.1f} mm, {limit}")
    lower = math.acos(max(-1.0, min(1.0, cos_knee)))  # >= 0: Knie zeigt nach hinten
    upper = math.atan2(x, down) - math.atan2(
        LOWER_LENGTH * math.sin(lower), UPPER_LENGTH + LOWER_LENGTH * math.cos(lower)
    )

    return LegAngles(math.degrees(shoulder), math.degrees(upper), math.degrees(lower))


def leg_points(angles: LegAngles) -> list[Vec3]:
    """Gelenkpunkte eines Beins in mm: Schulter, Hüftgelenk, Knie, Fuß."""
    shoulder, upper, lower = (math.radians(a) for a in angles)
    cos_s, sin_s = math.cos(shoulder), math.sin(shoulder)

    def to_leg_coords(forward: float, down: float) -> Vec3:
        # Punkt in der Beinebene -> Bein-Koordinaten (Ebene um die x-Achse gedreht)
        return (forward, HIP_LENGTH * cos_s + down * sin_s, HIP_LENGTH * sin_s - down * cos_s)

    knee = (UPPER_LENGTH * math.sin(upper), UPPER_LENGTH * math.cos(upper))
    foot = (
        knee[0] + LOWER_LENGTH * math.sin(upper + lower),
        knee[1] + LOWER_LENGTH * math.cos(upper + lower),
    )
    return [(0.0, 0.0, 0.0), to_leg_coords(0.0, 0.0), to_leg_coords(*knee), to_leg_coords(*foot)]


def leg_fk(angles: LegAngles) -> Vec3:
    """Gelenkwinkel (Grad) -> Fußposition (mm). Umkehrung von leg_ik()."""
    return leg_points(angles)[-1]


# ---------------------------------------------------------------------------
# Ganzer Körper
# ---------------------------------------------------------------------------
#
# Boden-Koordinaten in mm: Ursprung am Boden senkrecht unter der Körpermitte,
# x vorne, y links, z oben. Körper-Koordinaten: gleich, aber Ursprung in der
# Körpermitte (auf Höhe der Schulterachsen) und mit dem Körper mitgedreht.

LEGS = ("FL", "FR", "RL", "RR")  # gleiche Reihenfolge wie JOINT_ORDER in servos.py

# Schulterachsen in Körper-Koordinaten
SHOULDERS: dict[str, Vec3] = {
    "FL": (BODY_LENGTH / 2, BODY_WIDTH / 2, 0.0),
    "FR": (BODY_LENGTH / 2, -BODY_WIDTH / 2, 0.0),
    "RL": (-BODY_LENGTH / 2, BODY_WIDTH / 2, 0.0),
    "RR": (-BODY_LENGTH / 2, -BODY_WIDTH / 2, 0.0),
}


class BodyPose(NamedTuple):
    """Lage des Körpers. Winkel in Grad nach der Rechte-Hand-Regel."""
    height: float       # Höhe der Körpermitte über dem Boden (mm)
    roll: float = 0.0   # Drehung um x, + = linke Seite hoch
    pitch: float = 0.0  # Drehung um y, + = Nase runter
    yaw: float = 0.0    # Drehung um z, + = nach links gedreht


def _rotation(body: BodyPose) -> tuple[Vec3, Vec3, Vec3]:
    """Drehmatrix Körper -> Boden (erst roll, dann pitch, dann yaw)."""
    cr, sr = math.cos(math.radians(body.roll)), math.sin(math.radians(body.roll))
    cp, sp = math.cos(math.radians(body.pitch)), math.sin(math.radians(body.pitch))
    cy, sy = math.cos(math.radians(body.yaw)), math.sin(math.radians(body.yaw))
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _is_right(leg: str) -> bool:
    return leg[1] == "R"


def body_ik(feet: dict[str, Vec3], body: BodyPose) -> dict[str, LegAngles]:
    """Fußpunkte (Boden-Koordinaten, mm) -> Gelenkwinkel aller 4 Beine.

    Wirft ValueError, wenn ein Fuß nicht erreichbar ist.
    """
    rot = _rotation(body)
    angles = {}
    for leg in LEGS:
        fx, fy, fz = feet[leg]
        d = (fx, fy, fz - body.height)
        # Boden -> Körper: mit der transponierten Drehmatrix zurückdrehen
        px, py, pz = (sum(rot[r][c] * d[r] for r in range(3)) for c in range(3))
        sx, sy, sz = SHOULDERS[leg]
        x, y, z = px - sx, py - sy, pz - sz
        if _is_right(leg):
            y = -y  # Bein-Koordinaten: y zeigt immer nach außen
        try:
            angles[leg] = leg_ik(x, y, z)
        except ValueError as err:
            raise ValueError(f"{leg}: {err}") from None
    return angles


def body_points(angles: dict[str, LegAngles], body: BodyPose) -> dict[str, list[Vec3]]:
    """Gelenkpunkte aller Beine in Boden-Koordinaten: Schulter, Hüftgelenk, Knie, Fuß."""
    rot = _rotation(body)
    offset = (0.0, 0.0, body.height)
    result = {}
    for leg in LEGS:
        sx, sy, sz = SHOULDERS[leg]
        side = -1.0 if _is_right(leg) else 1.0
        points = []
        for x, y, z in leg_points(angles[leg]):
            p = (sx + x, sy + side * y, sz + z)  # Bein -> Körper
            points.append(tuple(sum(rot[r][c] * p[c] for c in range(3)) + offset[r] for r in range(3)))
        result[leg] = points
    return result


# ---------------------------------------------------------------------------
# Visualisierung (nur am PC)
# ---------------------------------------------------------------------------

def build_demo():
    """3D-Ansicht eines Beins mit Schiebereglern für die Fußposition."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-100, 250)
    ax.set_zlim(-260, 40)
    ax.set_box_aspect((400, 350, 300))  # gleicher Maßstab auf allen Achsen
    ax.view_init(elev=15, azim=-70)      # eher von der Seite, damit man das Knie sieht
    ax.set_xlabel("x vorne [mm]")
    ax.set_ylabel("y außen [mm]")
    ax.set_zlabel("z oben [mm]")

    ax.plot([-80, 80], [0, 0], [0, 0], "--", color="gray", label="Schulterachse")
    leg_line, = ax.plot([], [], [], "-o", lw=4, color="tab:blue", label="Bein")
    target, = ax.plot([], [], [], "x", ms=12, mew=3, color="tab:red", label="Ziel")
    ax.legend(loc="upper left")

    sliders = {}
    for i, (name, lo, hi, start) in enumerate([
        ("x", -200.0, 200.0, 0.0),
        ("y", -100.0, 250.0, HIP_LENGTH),
        ("z", -260.0, 0.0, -180.0),
    ]):
        slider_ax = fig.add_axes((0.2, 0.15 - i * 0.05, 0.6, 0.03))
        sliders[name] = Slider(slider_ax, f"{name} [mm]", lo, hi, valinit=start)

    def update(_=None) -> None:
        foot = tuple(sliders[n].val for n in "xyz")
        target.set_data_3d(*([v] for v in foot))
        try:
            angles = leg_ik(*foot)
        except ValueError as err:
            leg_line.set_visible(False)
            ax.set_title(str(err), color="tab:red")
        else:
            leg_line.set_data_3d(*zip(*leg_points(angles)))
            leg_line.set_visible(True)
            ax.set_title(
                f"Schulter {angles.shoulder:6.1f}°   Oberschenkel {angles.upper:6.1f}°   "
                f"Knie {angles.lower:6.1f}°",
                color="black",
            )
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    # Slider zurückgeben, sonst räumt Python sie weg und sie reagieren nicht mehr
    return fig, sliders


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _fig, _sliders = build_demo()
    plt.show()
