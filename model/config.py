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
COURSE_SIGMA = 10.0          # Grad Abweichung vom befohlenen Kurs
TORQUE_LIMIT = 0.95          # ab diesem Anteil des Blockiermoments zählt "am Anschlag"

# Gewichte der Belohnung (Plan-Tabelle 4.1). Die Terme selbst stehen in nova_env.py.
_STAGE_1 = {
    "tempo": 1.0,         # befohlenes Tempo halten
    "drehrate": 0.5,      # befohlene Drehrate halten (geradeaus: nicht abdriften)
    "kurs": 0.5,          # Richtung seit dem Start wie befohlen (bestraft auch langsames Abdriften)
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

# ---------------------------------------------------------------------------
# Training (Plan 4.1 und Phase 3), Stable-Baselines3 PPO
# ---------------------------------------------------------------------------

PPO_SETTINGS = {
    "n_steps": 1024,         # Schritte je Hund, bevor das Netz angepasst wird
    "batch_size": 2048,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,           # wie weit das Netz in die Zukunft denkt (ca. 100 Schritte = 2 s)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
NET_ARCH = [128, 128]        # 2 Schichten mit je 128 Neuronen, Actor und Critic getrennt
# Zufallsausschläge am Anfang (Standardabweichung e^-2 = 0,14). Gemessen ohne Training:
# mit -1,0 (Plan) läuft der Trab beim Ausprobieren nur 12 % der Strecke, mit -2,0 noch 79 %.
LOG_STD_INIT = -2.0
CHECKPOINT_EVERY = 1_000_000  # Schritte zwischen zwei Zwischenständen

# Lehrplan je Stufe: (Name, Millionen Schritte, Randomisierung, stärkster Stoß in N; 0 = keine Stöße).
# Erst ohne Zufall (lernt es überhaupt?), dann ein zufälliger Hund je Episode, dann Stöße.
# Stöße bis 25 N statt 15 N (Plan): Mit 15 N im Training hat das 1. Netz (19.11.2026) einen
# 20-N-Stoß nie erlebt und nicht abgefangen; 15 N übersteht schon der Trab ohne Netz.
CURRICULUM = {
    "residual": [
        ("1-ohne-zufall", 2.0, False, 0.0),
        ("2-randomisiert", 6.0, True, 0.0),
        ("3-mit-stoessen", 7.0, True, 25.0),
    ],
    # Stufe 2 startet mit Nachahmen (Behavior Cloning) und kommt in Phase 4.
}

# Gezieltes Stoßtraining (20.11.2026): ein fertiges Netz weitertrainieren, fast nur mit
# starken Seitenstößen (randomization.PUSH_STYLES). train.py --push-training <final.zip>.
# Vierter Wert: Name einer Stoß-Art statt stärkster Stoß in N.
PUSH_TRAINING = [("4-starke-seitenstoesse", 6.0, True, "stark-seitlich")]
