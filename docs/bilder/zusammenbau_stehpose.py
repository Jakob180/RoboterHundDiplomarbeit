"""Bild für docs/zusammenbau.md (aus dem Repo-Ordner starten: .venv/bin/python docs/bilder/zusammenbau_stehpose.py)."""
import math
import sys

sys.path[:0] = ["src"]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

from gait import STAND  # noqa: E402
from kinematics import HIP_LENGTH, LOWER_LENGTH, UPPER_LENGTH, body_ik, leg_points  # noqa: E402

INK, ACCENT, SOFT = "#1f2933", "#b5541c", "#7b8794"
a = body_ik(STAND.feet, STAND.body)["FL"]
pts = leg_points(a)  # Bein-Koordinaten: x vorne, y außen, z oben
hip, knee, foot = pts[1], pts[2], pts[3]

fig, (side, front) = plt.subplots(1, 2, figsize=(10, 5.6), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})

# --- Seitenansicht (linkes Bein, vorne = rechts) ---
side.plot([hip[0], knee[0], foot[0]], [hip[2], knee[2], foot[2]], color=INK, lw=7, solid_capstyle="round")
for p, label in ((hip, "Oberschenkel-Servo"), (knee, "Knie-Servo")):
    side.plot(p[0], p[2], "o", ms=16, color="white", mec=INK, mew=2.5, zorder=5)
side.plot(foot[0], foot[2], "o", ms=9, color=INK)
side.plot([0, 0], [10, -185], ls=(0, (4, 4)), color=SOFT, lw=1)
side.add_patch(Arc((0, 0), 90, 90, theta1=-90 - 50, theta2=-90, color=ACCENT, lw=2))
side.text(-15, -33, "50°", color=ACCENT, fontsize=12, ha="center", va="center")
# Kniewinkel
t1 = math.degrees(math.atan2(hip[2] - knee[2], hip[0] - knee[0]))
t2 = math.degrees(math.atan2(foot[2] - knee[2], foot[0] - knee[0]))
side.add_patch(Arc((knee[0], knee[2]), 50, 50, theta1=t2, theta2=t1, color=ACCENT, lw=2))
side.text(knee[0] + 30, knee[2] - 14, "90°", color=ACCENT, fontsize=12, va="center")
side.annotate("Knie zeigt\nnach hinten", (knee[0], knee[2]), (-175, -45), fontsize=10, color=INK,
              arrowprops=dict(arrowstyle="-", color=SOFT))
side.annotate("Fuß unter\nder Schulter", (foot[0], foot[2]), (45, -175), fontsize=10, color=INK,
              arrowprops=dict(arrowstyle="-", color=SOFT))
side.annotate("", (95, 25), (35, 25), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
side.text(65, 32, "vorne", ha="center", fontsize=10, color=INK)
side.set_title("Von der Seite (linkes Bein)", fontsize=12, color=INK, loc="left")
side.set_xlim(-190, 120)
side.set_ylim(-200, 55)

# --- Vorderansicht (linkes Bein, außen = links im Bild) ---
y0 = 0.0
front.plot([0, -HIP_LENGTH, -HIP_LENGTH], [0, 0, foot[2]], color=INK, lw=7, solid_capstyle="round")
front.plot(0, 0, "o", ms=16, color="white", mec=INK, mew=2.5, zorder=5)
front.plot(-HIP_LENGTH, 0, "o", ms=16, color="white", mec=INK, mew=2.5, zorder=5)
front.plot(-HIP_LENGTH, foot[2], "o", ms=9, color=INK)
front.plot([-HIP_LENGTH, -HIP_LENGTH], [10, -185], ls=(0, (4, 4)), color=SOFT, lw=1)
front.fill_between([0, 70], -14, 14, color="#e4e7eb", zorder=0)
front.text(38, 20, "Rumpf", fontsize=10, color=SOFT, ha="center")
front.annotate("Hüftstück\nwaagrecht", (-HIP_LENGTH / 2, 0), (-120, 40), fontsize=10, color=INK,
               arrowprops=dict(arrowstyle="-", color=SOFT))
front.annotate("Bein senkrecht", (-HIP_LENGTH, -100), (-150, -130), fontsize=10, color=INK,
               arrowprops=dict(arrowstyle="-", color=SOFT))
front.text(0, -26, "Schulter-\nServo", fontsize=9, color=INK, ha="center", va="top")
front.set_title("Von vorne (linkes Bein)", fontsize=12, color=INK, loc="left")
front.set_xlim(-170, 80)
front.set_ylim(-200, 55)

for ax in (side, front):
    ax.set_aspect("equal")
    ax.axhline(foot[2] - 10, color=SOFT, lw=1)
    ax.axis("off")
fig.suptitle("Alle Servos auf 90°: so werden die Hörner aufgesteckt (Stehpose)", fontsize=13, color=INK, x=0.02,
             ha="left")
fig.text(0.02, 0.02, f"Maße aus dem Modell (Platzhalter): Hüfte {HIP_LENGTH:.0f} mm, Oberschenkel {UPPER_LENGTH:.1f} mm, "
         f"Unterschenkel {LOWER_LENGTH:.0f} mm. Rechte Beine spiegelbildlich.", fontsize=9, color=SOFT)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig("docs/bilder/zusammenbau-stehpose.png")
print("ok", a)
