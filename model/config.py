"""
Alle Zahlen der Trainingsumgebung an einer Stelle (Plan Abschnitt 4.1 und 4.2).

Startwerte aus dem Plan. Beim Training werden vor allem die Gewichte der
Belohnung angepasst. Die Bereiche der Randomisierung stehen gesammelt in
randomization.py, die Skalierung von Ein- und Ausgängen in src/policy.py (die
braucht auch der Pi). Die PPO-Werte kommen mit train.py dazu (Phase 3).
"""

# Episode
EPISODE_SECONDS = 10.0       # 500 Schritte mit 50 Hz
WARMUP_SECONDS = 0.5         # vorher stehen, damit sich Hund und Lagefilter beruhigen

# Befehl: wechselt alle 2 bis 4 s
COMMAND_EVERY = (2.0, 4.0)   # s
COMMAND_SPEED = (0.0, 200.0)  # mm/s
COMMAND_YAW_RATE = 0.0       # ± °/s; erst geradeaus, später bis 28,6 (0,5 rad/s)

# Abbruch: umgefallen
FALL_HEIGHT = 80.0           # mm, Rumpfmitte tiefer
FALL_TILT = 40.0             # Grad Neigung

# Breite der Glockenkurven in der Belohnung
SPEED_SIGMA = 0.05           # m/s
YAW_SIGMA = 0.3              # rad/s
IMITATION_SIGMA = 0.3        # rad (alle 12 Gelenke zusammen)
TORQUE_LIMIT = 0.95          # ab diesem Anteil des Blockiermoments zählt "am Anschlag"

# Gewichte der Belohnung (Plan-Tabelle 4.1). Die Terme selbst stehen in nova_env.py.
_STAGE_1 = {
    "tempo": 1.0,         # befohlenes Tempo halten
    "drehrate": 0.5,      # befohlene Drehrate halten (geradeaus: nicht abdriften)
    "rutschen": -2.0,     # seitlich wegrutschen
    "neigung": -2.0,      # Rumpf schief
    "wackeln": -0.05,     # Drehrate um Längs- und Querachse
    "huepfen": -1.0,      # auf und ab
    "korrektur": -0.3,    # nur korrigieren, wenn es etwas bringt
    "ruckeln": -0.5,      # Servos und Getriebe schonen
    "servo_last": -0.2,   # Servos und Akku schonen
    "anschlag": -0.5,     # nichts lernen, was nur mit voller Kraft geht
    "umfallen": -10.0,    # einmalig beim Abbruch
}

REWARD = {
    "residual": {**_STAGE_1, "nachahmung": 0.0},
    # Stufe 2 (Plan 4.2): Die Ausgabe ist der ganze Gelenkwinkel, eine Strafe auf ihre
    # Größe würde das Laufen bestrafen. Nachahmen wird im Training später ausgeblendet
    # (1,0 -> 0,2). Fußrutschen und Fußhöhe kommen mit Phase 4.
    "full": {**_STAGE_1, "korrektur": 0.0, "ruckeln": -1.0, "nachahmung": 1.0},
}
