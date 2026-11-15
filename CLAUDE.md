# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projekt

Nova ist ein Roboterhund und unsere Diplomarbeit an der HTL Hallein (Teamprojekt). Die Mechanik ist ein Spot-Micro-Klon von Thingiverse mit 12 Servos (3 pro Bein), gesteuert von einem Raspberry Pi 5. Die Bewegung kommt von einem neuronalen Netz. Später soll Nova mit Akku laufen.

## Raspberry Pi 5

- `RPi.GPIO` und `pigpio` laufen am Pi 5 nicht, weil die GPIOs dort am RP1-Chip hängen. Stattdessen `gpiozero` oder `lgpio` verwenden.
- Raspberry Pi OS blockiert `pip install` außerhalb eines venv (PEP 668). Pakete immer in einem venv installieren, nie `sudo pip` oder `--break-system-packages` vorschlagen.
- Der Code läuft auf dem Pi (Linux, ARM64), wird aber auch unter Windows bearbeitet. Deshalb keine Windows-Pfade oder Windows-Befehle in Code und Skripten.

## Hardware-Sicherheit

- Servo-Winkel immer auf die Gelenkgrenzen begrenzen, bevor sie an die Hardware gehen. Das gilt auch für Ausgaben des neuronalen Netzes.
- Claude kann nicht am echten Roboter testen. Bei Code, der Servos bewegt, immer dazusagen, dass er noch nicht auf der Hardware getestet wurde.

## Arbeitsweise

- Zuerst lokal am PC/Laptop entwickeln und testen. Gepusht wird erst, wenn es passt (oder wenn jemand aus dem Team den Stand braucht).
- Claude committet und pusht nicht selbst, außer es wird ausdrücklich verlangt. Änderungen nur lokal machen, das Team prüft und pusht.
- Am Pi wird nicht programmiert, dort wird nur ausgeführt (siehe "Am Pi").
- Alles, was keine echte Hardware braucht (Kinematik, Gangmuster, Simulation, Netz), muss am PC mit `NOVA_SIM=1` laufen.

## Lokal testen (ohne Pi, ohne Servos)

Windows (cmd):

    set NOVA_SIM=1
    python src\servos.py

Windows (PowerShell): `$env:NOVA_SIM=1; python src/servos.py`
Linux / Pi: `NOVA_SIM=1 python3 src/servos.py`

Für Visualisierungen, Simulation und Tests braucht man am PC die Pakete aus `requirements-pc.txt` im venv (`.venv`, ist in `.gitignore`):
`python -m venv .venv`, dann `.venv\Scripts\python -m pip install -r requirements-pc.txt` (Linux: `.venv/bin/python ...`).
Starten z. B. mit `.venv\Scripts\python src/kinematics.py`, `src/gait.py`, `src/sim.py` oder `model/evaluate.py`.
Tests: `.venv\Scripts\python -m pytest` (laufen ohne Hardware, setzen `NOVA_SIM=1` selbst).

## Am Pi

Zwei getrennte Ordner, damit schnelle Tests nie mit Git kollidieren:

| Ordner | Wofür |
|---|---|
| `~/nova` | Git-Klon, der "fertige" Stand. Nur mit `git pull` aktualisieren, dort nichts von Hand ändern. Enthält das venv `~/nova/.venv`. |
| `~/nova-test` | Spielwiese für schnelle Tests: Code wird direkt vom Laptop per `scp` hineinkopiert, ohne Commit/Push. Nutzt das venv aus `~/nova`. |

- Schnell testen: Laptop -> `scp -r src <pi>:~/nova-test/` -> am Pi `~/nova/.venv/bin/python src/...` in `~/nova-test` starten.
- Fertig: am Laptop `git push`, am Pi in `~/nova` `git pull`.
- Am Pi ohne `NOVA_SIM` starten, dann bewegen sich die echten Servos.
- Eingerichtet seit 15.11.2026 (`~/nova` mit venv, `~/nova-test`, I2C an). Der Pi ist noch nicht im Hund, PCA9685 und GY-521 sind noch nicht angeschlossen.

## Aktueller Stand (Nov. 2026)

- Servos sind noch nicht gekauft, der Hund ist noch nicht 3D-gedruckt.
- `src/servos.py`: Servo-Schicht über PCA9685 (`adafruit-circuitpython-servokit`, I2C) mit Simulationsmodus.
  - Alle Bewegungen laufen über `Servos.set_joint()` / `Servos.set_all()`, dort wird auf die Gelenkgrenzen begrenzt; NaN/None -> Grundstellung.
  - Gelenknamen: `FL/FR/RL/RR` + `_shoulder/_upper/_lower`; `JOINT_ORDER` ist die Reihenfolge der 12 Netz-Ausgaben.
  - Kalibrierung: `Joint.zero` ist der Servo-Winkel beim Kinematik-Winkel 0. `to_servo_angle()` / `to_kinematic_angle()` rechnen um, `kinematic_limits()` liefert die Gelenkgrenzen als Kinematik-Winkel. Das ist die einzige Stelle für Gelenkgrenzen, die Simulation übernimmt sie.
  - `ANGLE_STEP`: kleinster Schritt des PCA9685 bei 50 Hz (0,44°).
  - Grenzwerte, Neutralstellungen, Nullpunkte, `inverted` und Pulsbreiten sind PLATZHALTER und müssen am echten Hund ausgemessen werden. Die Nullpunkte sind so gewählt, dass 90° ungefähr die Stehpose ist.
  - Im Fake-Modus getestet, noch nicht auf der Hardware.
- `src/kinematics.py`: Inverse Kinematik für ein Bein, `leg_ik(x, y, z)` -> `LegAngles(shoulder, upper, lower)`, Umkehrung `leg_fk()`. Einheiten mm und Grad.
  - Bein-Koordinaten: Ursprung auf der Schulterachse, x vorne, y nach AUSSEN, z oben. Dadurch rechnen alle 4 Beine gleich.
  - Ergibt Kinematik-Winkel (0/0/0 = Bein senkrecht gestreckt, Knie + = zeigt nach hinten), noch KEINE Servo-Winkel. Die Umrechnung macht `servos.to_servo_angle()`.
  - Beinlängen 55 / 107,5 / 130 mm und Körper 186 x 78 mm (Abstand der Schulterachsen) sind Spot-Micro-Platzhalter (mike4192/spot_micro_kinematics_python).
  - Nicht erreichbare Punkte -> `ValueError`. Gelenkgrenzen prüft erst `servos.py`.
  - `foot_ik(leg, foot, body)` rechnet ein einzelnes Bein (braucht `policy.py`, wenn nur ein Fuß nicht erreichbar ist).
  - Ganzer Körper: `body_ik(feet, BodyPose(height, roll, pitch, yaw))` -> `{"FL": LegAngles, ...}`, Zeichnen mit `body_points()`. Fußpunkte in Boden-Koordinaten: Ursprung am Boden unter der Körpermitte, x vorne, y LINKS, z oben. Winkel nach Rechte-Hand-Regel (pitch + = Nase runter).
  - `python src/kinematics.py` zeigt das Bein in 3D mit Schiebereglern für die Fußposition.
- `src/gait.py`: Grundposen `STAND`, `SIT` (Platzhalter-Werte), `blend()` für weiche Übergänge, `Trot`-Klasse (diagonale Paare, `update(dt, speed)` regelmäßig aufrufen, Standfüße rutschen nicht, Anlaufen/Anhalten ohne Sprung).
  - Trab-Standardwerte: 0,5 s Periode, 25 mm Schritthöhe, `duty` 0,6 (jeder Fuß 60 % der Zeit am Boden, beim Wechsel stehen kurz alle 4). Mit `duty` 0,5 war der Trab in der Simulation instabil.
  - Drehen: `update(dt, speed, yaw_rate)` mit `yaw_rate` in °/s (+ = Linkskurve). `trot.phase` (0..1), `trot.x`, `trot.y`, `trot.heading` sind öffentlich (fürs Netz).
  - `python src/gait.py` zeigt Stehen -> Sitzen -> Trab als 3D-Animation mit Winkelverlauf, `--gif datei.gif` speichert sie.
  - Trab mit 150 mm/s braucht bis ca. 300 °/s an Ober- und Unterschenkel. Mit den echten Servos vergleichen.
- `src/imu.py`: Komplementärfilter für den GY-521 (MPU6050). `ComplementaryFilter.update(gyro, accel, dt)` (rad/s, m/s²) -> Richtung "oben" im Rumpf, dazu `roll`/`pitch` in Grad. Läuft in der Simulation auf den MuJoCo-Werten und später am Pi. Das Auslesen über I2C fehlt noch (Sensor noch nicht angeschlossen).
- `src/nova_model.py`: MuJoCo-Modell, wird aus den Maßen in `kinematics.py` erzeugt. Gelenke heißen wie in `servos.py` und drehen wie die Kinematik-Winkel (0 = Bein senkrecht), die Winkel aus `body_ik()` gehen also direkt (in Radiant) an die Simulation.
  - Einfache Formen statt STL-Teile. Massen geschätzt, zusammen 2,6 kg, Akku und Powerbank als eigene Teile im Rumpf (Lage PLATZHALTER).
  - Servodaten aus dem Datenblatt bei 7,4 V: 2,59 N·m, 443 °/s. `SERVO_KP` und `SERVO_DEAD_TIME` sind PLATZHALTER, bis die Servos am Prüfstand gemessen sind.
  - Reibung und Weichheit des Kontakts kommen vom Boden (`priority`), Standard 0,8.
  - Sensoren: `gyro` und `accel` (Site `imu` im Rumpf), `rangefinder` vorne. `build_mjcf(self_collision=True)` lässt die Beine aneinander anstoßen.
  - Antriebe sind reine Drehmoment-Motoren, das Moment rechnet `sim.py`.
- `src/sim.py`: Physik-Simulation in MuJoCo (nur PC). Spielt den Ablauf aus `gait.py` mit 50 Hz ab.
  - Servo-Modell (`ServoParams`, `servo_torque()`): Zielwinkel begrenzt (`Servos.clamp`) und auf PCA9685-Stufen gerundet, Kalibrierfehler, Totzeit, Totzone, Moment = Kp·Fehler − Kd·Drehzahl, Kennlinie (Moment fällt mit der Drehzahl).
  - `sim.imu(delay)` (Drehrate, Beschleunigung wie beim MPU6050, auch verzögert), `sim.distance()` (mm oder None), `sim.push(kraft, dauer)` für Stöße, `sim.step(winkel, period)` für einen anderen Steuertakt.
  - `python src/sim.py` öffnet ein Fenster, `--headless` gibt nur die Auswertung aus (umgefallen?, Strecke, Neigung, Servo-Drehmoment). Trab-Werte mit `--speed`, `--period`, `--step-height`, `--duty`, `--self-collision`.
  - Ergebnis mit den Standardwerten: fällt nicht um, 90 % der geplanten Strecke (die Servos geben unter 2,6 kg etwas nach), max. ca. 1,5° Neigung, Servos max. ca. 1,6 N·m (61 %). Manche Werte (z. B. 0,6 s Periode) laufen deutlich schlechter, Änderungen am Gang immer erst hier prüfen.
  - Tempo am Laptop ca. 1.100 Steuerschritte/s (22-mal Echtzeit).
- `model/randomization.py`: Domain Randomization nach Plan Abschnitt 5.2 (Massen, Schwerpunkt, Boden, Servos, Totzeit, Kalibrierfehler, Spiel, Steuertakt, Stöße). `sample(rng)` würfelt, `apply(sim, r)` schreibt ins Modell. Fehler des Lagesensors (Rauschen, Nullpunkt, Einbauwinkel, Verzögerung) stehen in `ImuErrors` und wirken über `imu_reading()`. Unebener Boden fehlt noch.
- `model/evaluate.py`: Kennzahlen des klassischen Trabs in festen Szenarien und unter Randomisierung (feste Zufallszahlen). Ausgangswerte in `model/results/trab-phase1.json`, dauert am Laptop ca. 50 s.
  - Wichtigste Ergebnisse: Kalibrierfehler ±3° -> 41° Drehung, 2° Spiel -> nur 63 % Strecke, Stoß 15 N wird überstanden, ab 20 N fällt Nova um. Randomisiert im Mittel 73 % Strecke, 190 mm Drift, 29° Drehung.
- `src/policy.py`: Schnittstelle zum Netz, gleicher Code in der Trainingsumgebung und später am Pi. `mode.observe(oben, drehrate)` baut die 34 Eingänge, `mode.act(ausgabe, dt)` macht aus den 12 Ausgaben begrenzte Gelenkwinkel. NaN -> 0, alles auf -1..1 und auf die Gelenkgrenzen begrenzt. Noch nicht auf der Hardware getestet.
  - `ResidualMode` (Stufe 1): Trab + Korrektur der Fußpunkte (±20/±10/±15 mm). Ausgabe 0 = genau der klassische Trab.
  - `FullMode` (Stufe 2): Netz gibt die Gelenkwinkel vor (±30°, Schulter ±15°, höchstens 400 °/s), Trab läuft als Taktgeber und Vorbild (`reference`) mit.
  - Befehl mit `set_command(tempo, drehrate)`, wird weich angefahren.
- `model/config.py`: Zahlen der Trainingsumgebung (Episode 10 s, Befehle, Abbruch, Gewichte der Belohnung je Stufe).
- `model/nova_env.py`: Trainingsumgebung im Gymnasium-Format (`NovaEnv(mode="residual"|"full", randomize, pushes, max_yaw_rate)`). Würfelt bei jedem `reset()` einen neuen Hund, Belohnung nach Plan-Tabelle 4.1, Abbruch beim Umfallen.
  - Ca. 630 Schritte/s am Laptop. Korrektur 0 läuft nominal ohne Umfallen durch (87 % der befohlenen Strecke), ein zufälliges Netz fällt meist um.
- `tests/`: pytest für alle Module (ohne Hardware).

## Nächste Schritte

1. Inverse Kinematik für ein Bein: erledigt (`src/kinematics.py`). Offen: Beinlängen aus den STL-Dateien des Thingiverse-Modells abmessen.
2. Grundposen und Gang: erledigt (`src/gait.py`), mit Drehen. Kalibrierung Kinematik-Winkel -> Servo-Winkel steht in `servos.py` (Platzhalter-Nullpunkte). Offen: Nullpunkte und Grenzen am echten Hund ausmessen.
3. Simulation: Phase 1 aus `docs/neuronales-netz-plan.md` erledigt (Servo-Modell, IMU, Randomisierung, Auswertung). MuJoCo statt PyBullet, weil es PyBullet für Windows nicht als fertiges Paket gibt. Offen: echte Massen, Schwerpunkt und Servo-Messungen (Kp, Totzeit) eintragen.
4. Neuronales Netz per Reinforcement Learning in der Simulation am PC trainieren, als ONNX auf den Pi bringen. Der klassische Gang bleibt als Rückfallebene. Phase 2 (Trainingsumgebung) ist erledigt. Nächster Schritt ist Phase 3 im Plan: Training Stufe 1 am PC (`model/train.py` mit stable-baselines3 und torch).
