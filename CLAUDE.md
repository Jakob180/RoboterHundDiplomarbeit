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

Training (nur PC, Phase 3): zusätzlich PyTorch nur für die CPU und `model/requirements.txt` (Befehle stehen oben in der Datei). Dann:
`python model/train.py --quick` (Probelauf), `python model/train.py` (ganzer Lehrplan), Lernkurven mit `tensorboard --logdir runs`,
danach `python model/evaluate.py --policy checkpoints/<lauf>/final.zip` und `python model/export_onnx.py checkpoints/<lauf>/final.zip`.
`checkpoints/` und `runs/` sind in `.gitignore`, fertige Netze kommen nach `model/policies/` (werden committet). Seit 22.11.2026 leer: Die Netze v1 waren auf den alten Spot-Micro-Beinlängen trainiert und werden von `check_settings()` abgelehnt (Einstellungen enthalten jetzt `geometry`, `FORMAT_VERSION` 2). Neu trainieren (Stufe 1, dann `imitation.py`, dann Feinschliff) und als `nova_residual_v2` / `nova_full_v2` exportieren.

## Am Pi

Zwei getrennte Ordner, damit schnelle Tests nie mit Git kollidieren:

| Ordner | Wofür |
|---|---|
| `~/nova` | Git-Klon, der "fertige" Stand. Nur mit `git pull` aktualisieren, dort nichts von Hand ändern. Enthält das venv `~/nova/.venv`. |
| `~/nova-test` | Spielwiese für schnelle Tests: Code wird direkt vom Laptop per `scp` hineinkopiert, ohne Commit/Push. Nutzt das venv aus `~/nova`. |

- Schnell testen: Laptop -> `scp -r src <pi>:~/nova-test/` -> am Pi `~/nova/.venv/bin/python src/...` in `~/nova-test` starten.
- Fertig: am Laptop `git push`, am Pi in `~/nova` `git pull`.
- Am Pi ohne `NOVA_SIM` starten, dann bewegen sich die echten Servos.
- Eingerichtet seit 15.11.2026 (`~/nova` mit venv, `~/nova-test`, I2C an). Seit 21.11.2026 auch numpy, onnxruntime und smbus2 im venv (`requirements-pi.txt`). Der Pi ist noch nicht im Hund. Der PCA9685 hängt am Pi (Adresse 0x40), der erste Servo hat sich am 19.11.2026 bewegt. Der GY-521 ist noch nicht angeschlossen (Ersatz mit fertig gelöteten Pins kommt Mitte Dezember).

## Aktueller Stand (Nov. 2026)

- Teile sind da (15 Servos Miuzei MS62, PCA9685, 2S-LiPo 5200 mAh, Ladegerät), der 3D-Druck läuft.
- **Betreuer, 20.11.2026:** Stufe 2 ist Pflicht (nicht nur Soll). Am echten Hund muss das Netz besser laufen als der Trab. Neu als Pflicht: Sicherheits-Erkennung (welcher Motor zu wenig/zu viel Strom bekommt, Abschalten bei Kurzschluss). Abnahme Phase 3 neu, siehe `evaluate.py`. Details in `docs/neuronales-netz-plan.md` (3.2, 4.4, Phase 3).
- `src/servos.py`: Servo-Schicht über PCA9685 (`adafruit-circuitpython-servokit`, I2C) mit Simulationsmodus.
  - Alle Bewegungen laufen über `Servos.set_joint()` / `Servos.set_all()`, dort wird auf die Gelenkgrenzen begrenzt; NaN/None -> Grundstellung.
  - Gelenknamen: `FL/FR/RL/RR` + `_shoulder/_upper/_lower`; `JOINT_ORDER` ist die Reihenfolge der 12 Netz-Ausgaben.
  - Kalibrierung: `Joint.zero` ist der Servo-Winkel beim Kinematik-Winkel 0. `to_servo_angle()` / `to_kinematic_angle()` rechnen um, `kinematic_limits()` liefert die Gelenkgrenzen als Kinematik-Winkel. Das ist die einzige Stelle für Gelenkgrenzen, die Simulation übernimmt sie.
  - `ANGLE_STEP`: kleinster Schritt des PCA9685 bei 50 Hz (0,44°).
  - Grenzwerte, Neutralstellungen, Nullpunkte, `inverted` und Pulsbreiten sind PLATZHALTER und müssen am echten Hund ausgemessen werden. Die Nullpunkte sind so gewählt, dass 90° ungefähr die Stehpose ist (Bein senkrecht, Oberschenkel 50° nach hinten, Knie 90°).
  - Kalibrierung: `src/calibrate.py --hardware` misst am aufgebockten Hund je Gelenk Nullpunkt (Bezugslage), Richtung und Anschläge, speichert `~/nova-kalibrierung.json`. Als `calibration.json` im Repo-Ordner überschreibt sie die Platzhalter (`servos.load_calibration`, Pfad auch über `NOVA_CALIBRATION`; die Tests setzen ihn auf eine fehlende Datei). `Servos.set_raw()` fährt ohne Gelenkgrenzen (nur 5..175°) und ist NUR für die Kalibrierung. Anleitung mit Bild: `docs/zusammenbau.md`.
  - Im Fake-Modus getestet, noch nicht auf der Hardware.
- `src/kinematics.py`: Inverse Kinematik für ein Bein, `leg_ik(x, y, z)` -> `LegAngles(shoulder, upper, lower)`, Umkehrung `leg_fk()`. Einheiten mm und Grad.
  - Bein-Koordinaten: Ursprung auf der Schulterachse, x vorne, y nach AUSSEN, z oben. Dadurch rechnen alle 4 Beine gleich.
  - Ergibt Kinematik-Winkel (0/0/0 = Bein senkrecht gestreckt, Knie + = zeigt nach hinten), noch KEINE Servo-Winkel. Die Umrechnung macht `servos.to_servo_angle()`.
  - Beinlängen seit 22.11.2026 gemessen ("SM2"-Beine): Oberschenkel 125 mm, Unterschenkel 142 mm bis zur Fußsohle (122 mm bis zur Mitte des Fußrings + 20 mm runder Fuß). Hüftversatz 55 mm und Körper 186 x 78 mm (Abstand der Schulterachsen) sind Spot-Micro-Maße (die Breite stimmt mit der Schulterplatte überein); am fertigen Hund nachmessen.
  - Nicht erreichbare Punkte -> `ValueError`. Gelenkgrenzen prüft erst `servos.py`.
  - `foot_ik(leg, foot, body)` rechnet ein einzelnes Bein (braucht `policy.py`, wenn nur ein Fuß nicht erreichbar ist).
  - Ganzer Körper: `body_ik(feet, BodyPose(height, roll, pitch, yaw))` -> `{"FL": LegAngles, ...}`, Zeichnen mit `body_points()`. Fußpunkte in Boden-Koordinaten: Ursprung am Boden unter der Körpermitte, x vorne, y LINKS, z oben. Winkel nach Rechte-Hand-Regel (pitch + = Nase runter).
  - `python src/kinematics.py` zeigt das Bein in 3D mit Schiebereglern für die Fußposition.
- `src/gait.py`: Grundposen `STAND`, `SIT` (Platzhalter-Werte), `blend()` für weiche Übergänge, `Trot`-Klasse (diagonale Paare, `update(dt, speed)` regelmäßig aufrufen, Standfüße rutschen nicht, Anlaufen/Anhalten ohne Sprung).
  - Trab-Standardwerte: 0,5 s Periode, 25 mm Schritthöhe, `duty` 0,6 (jeder Fuß 60 % der Zeit am Boden, beim Wechsel stehen kurz alle 4). Mit `duty` 0,5 war der Trab in der Simulation instabil.
  - Drehen: `update(dt, speed, yaw_rate)` mit `yaw_rate` in °/s (+ = Linkskurve). `trot.phase` (0..1), `trot.x`, `trot.y`, `trot.heading` sind öffentlich (fürs Netz). `trot.swing(bein)` sagt, wie weit ein Fuß im Schwung ist (0..1, None = steht).
  - `python src/gait.py` zeigt Stehen -> Sitzen -> Trab als 3D-Animation mit Winkelverlauf, `--gif datei.gif` speichert sie.
  - Trab mit 150 mm/s braucht bis ca. 300 °/s an Ober- und Unterschenkel. Mit den echten Servos vergleichen.
- `src/imu.py`: Komplementärfilter für den GY-521 (MPU6050). `ComplementaryFilter.update(gyro, accel, dt)` (rad/s, m/s²) -> Richtung "oben" im Rumpf, dazu `roll`/`pitch` in Grad. Läuft in der Simulation auf den MuJoCo-Werten und am Pi.
- `src/mpu6050.py`: MPU6050 über I2C (smbus2, Adresse 0x68), ±500 °/s, ±4 g, Tiefpass 44 Hz. `read()` -> (Drehrate rad/s, Beschleunigung m/s²) in Rumpf-Achsen, `calibrate()` misst den Nullpunkt des Gyroskops (Hund muss still sein). Nimmt auch gängige Nachbau-Chips (WHO_AM_I 0x68/0x70/0x72/0x98). Einbaulage `MOUNT` ist PLATZHALTER. Mit nachgebautem Bus getestet, noch nicht am echten Sensor.
- `src/controller.py`: Hauptschleife für den Pi (Plan 5.4/5.6), 50 Hz: Lagesensor -> Filter -> Netz (ONNX) -> Gelenkwinkel -> Servos, CSV-Protokoll je Takt. `k` (0..1) blendet zwischen Trab und Netz (Standard 0). Sicherheit: Neigung > 30° -> Servos locker; NaN vom Netz, Sensor-Fehler oder > 3 Takte zu spät -> Takt ohne Netz; 0,5 s ohne Sensor -> anhalten. Aufstehen aus der Liegepose (`gait.LIE`, 80 mm), Hinlegen, dann locker. Hardware: `PiHardware` (Servos + MPU6050), `SimHardware` (MuJoCo, Software-in-the-Loop), `DryHardware` (Rechenzeit messen). Echte Servos nur mit `--hardware`. `NOVA_SIM=1 python src/controller.py --policy model/policies/nova_full_v1.onnx --k 1 --speed 150`, `--dry` misst die Rechenzeit (Laptop 2,2 ms, Pi 5 0,3 ms pro Takt, ohne I2C). Beide Netze laufen in der Schleife wie in der Trainingsumgebung (Test). Noch nicht auf der Hardware getestet.
- Am Pi: Pakete aus `requirements-pi.txt` ins venv (`~/nova/.venv/bin/python -m pip install -r requirements-pi.txt`).
- `src/nova_model.py`: MuJoCo-Modell, wird aus den Maßen in `kinematics.py` erzeugt. Gelenke heißen wie in `servos.py` und drehen wie die Kinematik-Winkel (0 = Bein senkrecht), die Winkel aus `body_ik()` gehen also direkt (in Radiant) an die Simulation.
  - Einfache Formen statt STL-Teile. Massen seit 22.11.2026 aus gewogenen Teilen, zusammen 2,1 kg (Knieservo im Unterschenkel, Hüftservo am Hüftstück; Powerbank, Schrauben, Lager, Kabel geschätzt). Akku und Powerbank als eigene Teile im Rumpf (Lage PLATZHALTER). Fuß: Kugel mit 20 mm Radius.
  - Servodaten aus dem Datenblatt bei 7,4 V: 2,59 N·m, 443 °/s. `SERVO_KP` und `SERVO_DEAD_TIME` sind PLATZHALTER, bis die Servos am Prüfstand gemessen sind.
  - Reibung und Weichheit des Kontakts kommen vom Boden (`priority`), Standard 0,8.
  - Sensoren: `gyro` und `accel` (Site `imu` im Rumpf), `rangefinder` vorne. `build_mjcf(self_collision=True)` lässt die Beine aneinander anstoßen.
  - Antriebe sind reine Drehmoment-Motoren, das Moment rechnet `sim.py`.
- `src/sim.py`: Physik-Simulation in MuJoCo (nur PC). Spielt den Ablauf aus `gait.py` mit 50 Hz ab.
  - Servo-Modell (`ServoParams`, `servo_torque()`): Zielwinkel begrenzt (`Servos.clamp`) und auf PCA9685-Stufen gerundet, Kalibrierfehler, Totzeit, Totzone, Moment = Kp·Fehler − Kd·Drehzahl, Kennlinie (Moment fällt mit der Drehzahl).
  - `sim.imu(delay)` (Drehrate, Beschleunigung wie beim MPU6050, auch verzögert), `sim.distance()` (mm oder None), `sim.push(kraft, dauer)` für Stöße, `sim.step(winkel, period)` für einen anderen Steuertakt.
  - `python src/sim.py` öffnet ein Fenster, `--headless` gibt nur die Auswertung aus (umgefallen?, Strecke, Neigung, Servo-Drehmoment). Trab-Werte mit `--speed`, `--period`, `--step-height`, `--duty`, `--self-collision`.
  - Ergebnis mit den Standardwerten (Maße vom 22.11.2026, Stehhöhe 190 mm): fällt nicht um, 92 % der geplanten Strecke, max. ca. 2° Neigung, Servos max. ca. 1,3 N·m (51 %). Manche Werte (z. B. 0,6 s Periode) laufen deutlich schlechter, Änderungen am Gang immer erst hier prüfen.
  - Tempo am Laptop ca. 1.100 Steuerschritte/s (22-mal Echtzeit).
- `model/randomization.py`: Domain Randomization nach Plan Abschnitt 5.2 (Massen, Schwerpunkt, Boden, Servos, Totzeit, Kalibrierfehler, Spiel, Steuertakt, Stöße). `sample(rng)` würfelt, `apply(sim, r)` schreibt ins Modell. Fehler des Lagesensors (Rauschen, Nullpunkt, Einbauwinkel, Verzögerung) stehen in `ImuErrors` und wirken über `imu_reading()`. Unebener Boden fehlt noch.
- `model/evaluate.py`: Kennzahlen des klassischen Trabs in festen Szenarien und unter Randomisierung (feste Zufallszahlen). Ausgangswerte in `model/results/trab-phase1.json`, dauert am Laptop ca. 50 s.
  - Wichtigste Ergebnisse (22.11.2026, gemessene Maße): Kalibrierfehler ±3° -> 58° Drehung, 2° Spiel -> nur 61 % Strecke, Stoß 10 N wird überstanden, ab 15 N fällt Nova um (höher und leichter als vorher). Randomisiert im Mittel 80 % Strecke, 155 mm Drift, 24° Drehung; mit Stößen fallen 2 von 20.
- `src/policy.py`: Schnittstelle zum Netz, gleicher Code in der Trainingsumgebung und später am Pi. `mode.observe(oben, drehrate)` baut die 34 Eingänge, `mode.act(ausgabe, dt)` macht aus den 12 Ausgaben begrenzte Gelenkwinkel. NaN -> 0, alles auf -1..1 und auf die Gelenkgrenzen begrenzt. Noch nicht auf der Hardware getestet.
  - `ResidualMode` (Stufe 1): Trab + Korrektur der Fußpunkte (Standard ±20/±10/±15 mm, je Netz einstellbar mit `ResidualMode(foot_scale=...)`, seitlich gehen bis ±40 mm). Ausgabe 0 = genau der klassische Trab.
  - `FullMode` (Stufe 2): Netz gibt die Gelenkwinkel vor (±30°, Schulter ±15°, höchstens 400 °/s), Trab läuft als Taktgeber und Vorbild (`reference`) mit.
  - Befehl mit `set_command(tempo, drehrate)`, wird weich angefahren.
  - `OnnxNetwork(pfad)` führt ein fertiges Netz aus (braucht nur onnxruntime). `settings(mode)` beschreibt alle Skalierungen; beim Training als `settings.json` gespeichert, beim Export als `.json` neben das Netz. `check_settings()` verweigert Netze, die mit anderen Einstellungen trainiert wurden (außer der Fußkorrektur: `mode_from_settings()` baut die Stufe so, wie das Netz trainiert wurde).
- `model/config.py`: Zahlen der Trainingsumgebung (Episode 10 s, Befehle, Abbruch, Gewichte der Belohnung je Stufe).
- `model/nova_env.py`: Trainingsumgebung im Gymnasium-Format (`NovaEnv(mode="residual"|"full", randomize, pushes, max_yaw_rate)`). Würfelt bei jedem `reset()` einen neuen Hund, Belohnung nach Plan-Tabelle 4.1 plus Term „kurs“ (Abweichung vom befohlenen Kurs, Breite 10°, weil langsames Abdriften sonst kaum kostet), Abbruch beim Umfallen.
  - Ca. 630 Schritte/s am Laptop. Korrektur 0 läuft nominal ohne Umfallen durch (87 % der befohlenen Strecke), ein zufälliges Netz fällt meist um.
  - Stufe 2 (`mode="full"`): Selbstkollision an (Beine können aneinanderstoßen), zusätzliche Terme `fuss_rutschen` (Geschwindigkeit der Füße am Boden) und `fusshoehe` (Fuß mitten im Schwung unter 15 mm, gleitend; der Trab hebt in der Simulation im Mittel nur 13 mm). Gewichte je Umgebung, `env.set_reward_weight()` (z. B. Nachahmung ausblenden).
  - Feste Szenarien für Vergleiche: `env.reset(seed=0, options={"randomization": r, "command": (150.0, 0.0)})`. Am Episodenende steht in `info["nova"]` die gelaufene Strecke und ob Nova umgefallen ist.
- `model/train.py`: PPO-Training (Stable-Baselines3) nach Lehrplan aus `config.CURRICULUM`: ohne Zufall (2 Mio.), randomisiert (6 Mio.), mit Stößen bis 25 N (7 Mio. Schritte). Am PC (Ryzen 5 5600X, 12 Hunde) ca. 3.000 Schritte/s, also ca. 1,5 h. So viele Hunde parallel wie CPU-Threads. Anfangs-Zufall `LOG_STD_INIT` -2,0 statt -1,0 aus dem Plan (mit -1,0 läuft der Trab beim Ausprobieren nur 12 % der Strecke). Zwischenstände in `checkpoints/<lauf>/`, Lernkurven inkl. aller Belohnungs-Terme in `runs/`. `--foot-scale 20 40 15` trainiert mit ±40 mm seitlicher Fußkorrektur (steht dann in `settings.json`). `--start-from checkpoints/full-bc/final.zip` macht den PPO-Feinschliff von Stufe 2 (`config.CURRICULUM["full"]`: 0,5 Mio. Schritte nur Critic, dann 20 Mio. mit Stößen bis 15 N (der frühere Abschnitt mit 25 N hat das Netz verschlechtert), Nachahmung 1,0 -> 0,2 ausgeblendet, Zufall e^-3 statt e^-2, weil er direkt auf die Gelenke wirkt). `--push-training checkpoints/<lauf>/final.zip` trainiert ein fertiges Netz mit `config.PUSH_TRAINING` weiter (6 Mio. Schritte, fast nur Seitenstöße 12 bis 28 N, `randomization.PUSH_STYLES`), Modus und Fußkorrektur kommen vom Netz. Stufe 2 kommt in Phase 4.
- `model/imitation.py` (Stufe 2, Schritte 1 und 2 aus Plan 4.2): `Teacher` (Trab, optional mit Netz aus Stufe 1, läuft im Gleichschritt mit der Umgebung), `collect()` sammelt Paare aus Beobachtung (Stufe 2) und Lehrer-Ausgabe (parallel, ausgeführt mit etwas Rauschen), `behavior_cloning()` trainiert den PPO-Actor darauf und speichert `final.zip` + `settings.json` wie ein trainiertes Netz. `python model/imitation.py --teacher checkpoints/<lauf>/final.zip` (1 Mio. Paare). Ergebnis 20.11.2026 am Laptop mit Trab als Lehrer (300.000 Paare, 5 min): Nova läuft in der Simulation allein mit dem Netz, genau wie der Trab (95 % Strecke, Stoßgrenze 17/19 N). `policy.angles_to_full_action()` rechnet Gelenkwinkel in Ausgaben von Stufe 2 um, `env.last_imu` gibt den letzten Lagesensor.
- `model/export_onnx.py`: `final.zip` -> `model/policies/<name>.onnx` + `.json`, vergleicht danach Original und ONNX (braucht onnxruntime, gibt es für Python 3.15 unter Windows noch nicht -> am Laptop machen).
- `model/evaluate.py --policy <.zip|.onnx>`: Trab ohne Netz gegen Trab mit Netz auf denselben Hunden (Szenarien + 20 randomisierte Episoden mit Stößen, 150 mm/s) und Stoßgrenze nach links/rechts (`push_limit()`, stärkster Seitenstoß in 1-N-Schritten ohne Umfallen; Trab mit den gemessenen Maßen: 12 N links, 13 N rechts, vorher 17/19 N). Hakt die Abnahme ab (mit dem Betreuer am 20.11.2026 vereinbart): Strecke ≥ 90 % und nicht öfter umgefallen, Kurs halb so stark abgedreht, Stoßgrenze beidseitig ≥ 1,25 × Trab, Kalibrierfehler halb so stark abgedreht.
  - Ergebnisse 19. bis 21.11.2026 (noch mit den alten Spot-Micro-Maßen): Netz 2 (Stöße bis 25 N, ±10 mm) läuft 96 % statt 81 % der Strecke, dreht 17° statt 40° ab, Kalibrierfehler 16° statt 51°. Den 20-N-Stoß übersteht es nicht (mit ±10 mm physikalisch nicht machbar), daher das Training mit ±40 mm.
- `tests/`: pytest für alle Module (ohne Hardware).

## Nächste Schritte

1. Inverse Kinematik für ein Bein: erledigt (`src/kinematics.py`). Offen: Beinlängen aus den STL-Dateien des Thingiverse-Modells abmessen.
2. Grundposen und Gang: erledigt (`src/gait.py`), mit Drehen. Kalibrierung Kinematik-Winkel -> Servo-Winkel steht in `servos.py` (Platzhalter-Nullpunkte). Offen: Nullpunkte und Grenzen am echten Hund ausmessen.
3. Simulation: Phase 1 aus `docs/neuronales-netz-plan.md` erledigt (Servo-Modell, IMU, Randomisierung, Auswertung). MuJoCo statt PyBullet, weil es PyBullet für Windows nicht als fertiges Paket gibt. Offen: echte Massen, Schwerpunkt und Servo-Messungen (Kp, Totzeit) eintragen.
4. Neuronales Netz per Reinforcement Learning in der Simulation am PC trainieren, als ONNX auf den Pi bringen. Der klassische Gang bleibt als Rückfallebene. Phase 2 (Trainingsumgebung) ist erledigt, Phase 3 läuft: zwei Trainings am PC fertig (Netz 2 erfüllt 3 von 4 Kriterien der neuen Abnahme, Stoßgrenze noch nicht gemessen). Nächster Schritt: 3. Training mit `--foot-scale 20 40 15`, dann Netz 2 und 3 mit `evaluate.py --policy` vergleichen und das bessere als ONNX exportieren (am Laptop, wegen onnxruntime).
5. Sicherheits-Erkennung (Betreuer): Sicherung gegen Kurzschluss, Not-Aus, Strommessung (Variante 1: ein INA226 + Selbsttest je Motor; Variante 2: Strom je Servo). Hardware-Variante noch offen, Software kommt mit `src/controller.py` (Phase 4) und wird gegen die Simulation getestet.
