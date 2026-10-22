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

Für die Visualisierungen braucht man am PC matplotlib im venv (`.venv`, ist in `.gitignore`):
`python -m venv .venv`, dann `.venv\Scripts\python -m pip install matplotlib` und `.venv\Scripts\python src/kinematics.py`.

## Am Pi

Zwei getrennte Ordner, damit schnelle Tests nie mit Git kollidieren:

| Ordner | Wofür |
|---|---|
| `~/nova` | Git-Klon, der "fertige" Stand. Nur mit `git pull` aktualisieren, dort nichts von Hand ändern. Enthält das venv `~/nova/.venv`. |
| `~/nova-test` | Spielwiese für schnelle Tests: Code wird direkt vom Laptop per `scp` hineinkopiert, ohne Commit/Push. Nutzt das venv aus `~/nova`. |

- Schnell testen: Laptop -> `scp -r src <pi>:~/nova-test/` -> am Pi `~/nova/.venv/bin/python src/...` in `~/nova-test` starten.
- Fertig: am Laptop `git push`, am Pi in `~/nova` `git pull`.
- Am Pi ohne `NOVA_SIM` starten, dann bewegen sich die echten Servos.
- Noch nicht eingerichtet (der Pi ist noch nicht im Hund). Einmalig: `git clone ... ~/nova`, venv in `~/nova/.venv` anlegen, I2C aktivieren, `mkdir ~/nova-test`.

## Aktueller Stand (Okt. 2026)

- Servos sind noch nicht gekauft, der Hund ist noch nicht 3D-gedruckt.
- `src/servos.py`: Servo-Schicht über PCA9685 (`adafruit-circuitpython-servokit`, I2C) mit Simulationsmodus.
  - Alle Bewegungen laufen über `Servos.set_joint()` / `Servos.set_all()`, dort wird auf die Gelenkgrenzen begrenzt; NaN/None -> Grundstellung.
  - Gelenknamen: `FL/FR/RL/RR` + `_shoulder/_upper/_lower`; `JOINT_ORDER` ist die Reihenfolge der 12 Netz-Ausgaben.
  - Grenzwerte, Neutralstellungen, `inverted` und Pulsbreiten sind PLATZHALTER und müssen am echten Hund ausgemessen werden.
  - Im Fake-Modus getestet, noch nicht auf der Hardware.
- `src/kinematics.py`: Inverse Kinematik für ein Bein, `leg_ik(x, y, z)` -> `LegAngles(shoulder, upper, lower)`, Umkehrung `leg_fk()`. Einheiten mm und Grad.
  - Bein-Koordinaten: Ursprung auf der Schulterachse, x vorne, y nach AUSSEN, z oben. Dadurch rechnen alle 4 Beine gleich.
  - Ergibt Kinematik-Winkel (0/0/0 = Bein senkrecht gestreckt, Knie + = zeigt nach hinten), noch KEINE Servo-Winkel. Die Umrechnung kommt mit der Kalibrierung.
  - Beinlängen 55 / 107,5 / 130 mm und Körper 186 x 78 mm (Abstand der Schulterachsen) sind Spot-Micro-Platzhalter (mike4192/spot_micro_kinematics_python).
  - Nicht erreichbare Punkte -> `ValueError`. Gelenkgrenzen prüft erst `servos.py`.
  - Ganzer Körper: `body_ik(feet, BodyPose(height, roll, pitch, yaw))` -> `{"FL": LegAngles, ...}`, Zeichnen mit `body_points()`. Fußpunkte in Boden-Koordinaten: Ursprung am Boden unter der Körpermitte, x vorne, y LINKS, z oben. Winkel nach Rechte-Hand-Regel (pitch + = Nase runter).
  - `python src/kinematics.py` zeigt das Bein in 3D mit Schiebereglern für die Fußposition.
- `src/gait.py`: Grundposen `STAND`, `SIT` (Platzhalter-Werte), `blend()` für weiche Übergänge, `Trot`-Klasse (diagonale Paare, `update(dt, speed)` regelmäßig aufrufen, Standfüße rutschen nicht, Anlaufen/Anhalten ohne Sprung).
  - `python src/gait.py` zeigt Stehen -> Sitzen -> Trab als 3D-Animation mit Winkelverlauf, `--gif datei.gif` speichert sie.
  - Trab mit 150 mm/s und 0,5 s Periode braucht bis ca. 250 °/s an Ober- und Unterschenkel. Mit den echten Servos vergleichen.

## Nächste Schritte

1. Inverse Kinematik für ein Bein: erledigt (`src/kinematics.py`). Offen: Beinlängen aus den STL-Dateien des Thingiverse-Modells abmessen.
2. Grundposen und Gang: erledigt (`src/gait.py`). Offen: Umrechnung Kinematik-Winkel -> Servo-Winkel (Kalibrierung), Gelenkgrenzen schon in der Gangplanung berücksichtigen.
3. Simulation in PyBullet mit Spot-Micro-URDF.
4. Neuronales Netz per Reinforcement Learning in der Simulation am PC trainieren, als ONNX auf den Pi bringen. Der klassische Gang bleibt als Rückfallebene.
