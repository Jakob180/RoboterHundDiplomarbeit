# Neuronales Netz für Nova: Analyse und Plan

Stand: 13.11.2026, überarbeitet mit euren Antworten (Servo-Datenblatt, MPU6050,
Abgabe ca. April 2027, Training am PC). Nur Analyse und Plan, am Code wurde
nichts geändert. Die Messwerte in Abschnitt 2 stammen aus der vorhandenen
Simulation (`src/sim.py`, MuJoCo 4.0.1) mit einem Hilfsskript außerhalb des
Repos. Gerechnet ist mit den Platzhalter-Werten, für den echten Hund gelten
sie noch nicht.

## Kurzfassung

- **Die Simulation ist eine echte Physik-Simulation** (MuJoCo: Schwerkraft,
  Bodenkontakt, Reibung, Servos mit begrenzter Kraft). Als Grundlage fürs
  Training taugt sie. Vorher muss aber einiges ergänzt oder korrigiert werden:
  IMU und Ultraschall fehlen im Modell, die Fußreibung wirkt gar nicht, die
  Gelenkbereiche passen nicht zu 180°-Servos, und Servo-Verzögerung,
  Moment-Drehzahl-Kennlinie und Spiel fehlen.
- **Empfehlung: zwei Stufen mit demselben Unterbau.**
  - **Stufe 1, Mindestziel: Das Netz verbessert den Trab (Residual-RL).**
    Der klassische Trab mit inverser Kinematik bleibt, ein kleines Netz (PPO,
    Stable-Baselines3) lernt Korrekturen der Fußpunkte, damit Nova mit der IMU
    das Gleichgewicht hält, geradeaus läuft und Stöße abfängt.
  - **Stufe 2, Hauptziel: Das Netz läuft selbst.** Es gibt alle 12
    Gelenkwinkel selbst aus. Es lernt zuerst, den Trab nachzumachen (Behavior
    Cloning), und wird dann mit RL besser (PPO mit Nachahmungs-Belohnung, die
    langsam ausgeblendet wird). Das ist deutlich sicherer als von null zu
    lernen.
  - Der klassische Trab bleibt in beiden Stufen die Rückfallebene. Der
    Ultraschall kommt als Kann-Ziel darüber, zuerst als Regeln.
- **Eure Servos passen:** 4,8 bis 8,4 V, also direkt am 2S-LiPo. Blockiermoment
  2,5 bis 2,7 N·m, 430 bis 460 °/s. Bei der Stromversorgung ist aber Vorsicht
  nötig (bis 3 A pro Servo).
- **Euer MPU6050 reicht** fürs Netz. Wichtig ist, in der Simulation dieselben
  Rohwerte und denselben Lagefilter zu verwenden wie am Pi.
- **Training auf deinem PC mit der CPU.** Die Grafikkarte spielt keine Rolle.
  Das fertige Netz kommt als `.onnx` mit `onnxruntime` auf den Pi, eine
  Rechnung dauert dort weit unter 1 ms.
- **Kritischer Termin:** Der Hund sollte bis **Ende Jänner 2027** gebaut sein,
  sonst bleibt zu wenig Zeit für die Tests am echten Hund (Zeitplan in
  Abschnitt 6).

---

## 1. Bestandsaufnahme: Was im Repo ist

### 1.1 Struktur

```
RoboterHundDiplomarbeit/
├── CLAUDE.md           Arbeitsweise, Pi-Regeln, Stand
├── README.md           Projektüberblick (Ordner docs/, hardware/, model/ sind geplant, gibt es noch nicht)
├── .gitignore          ignoriert checkpoints/, runs/, *.log; .onnx/.pt bewusst NICHT
└── src/
    ├── servos.py       135 Zeilen  Servo-Schicht (PCA9685) mit Simulationsmodus
    ├── kinematics.py   269 Zeilen  Inverse Kinematik Bein + Körper, Visualisierung
    ├── gait.py         261 Zeilen  Grundposen, Übergänge, Trab, Animation
    ├── nova_model.py   135 Zeilen  MuJoCo-Modell (MJCF) aus den Maßen erzeugt
    └── sim.py          168 Zeilen  Physik-Simulation, Fenster oder Auswertung
```

Keine Tests, keine `requirements.txt`, kein Paket (die Module importieren sich
gegenseitig flach, z. B. `from kinematics import ...`, laufen also aus `src/`).

### 1.2 Simulation

- **Engine:** MuJoCo (Python-Paket `mujoco`), gewählt statt PyBullet, weil es
  PyBullet für Windows nicht fertig gibt.
- **Echte Physik, nicht nur Visualisierung.** Starrkörperdynamik mit freiem
  Rumpf (`freejoint`), Schwerkraft, Bodenkontakt mit Reibung, Zeitschritt 2 ms
  (`implicitfast`). Die Animation in `gait.py` (matplotlib) ist dagegen
  reine Kinematik ohne Physik.
- **Ablauf:** `sim.py` spielt die fertig berechneten Frames aus
  `gait.demo_frames()` ab (Stehen, Sitzen, Aufstehen, Trab, Anhalten). Steuerung
  mit 50 Hz, pro Steuerschritt 10 Physikschritte.
- **Servos:** MuJoCo-`position`-Aktuatoren (P-Regler, `kp` = 30 N·m/rad,
  Kraft begrenzt auf ±2,0 N·m). Die Geschwindigkeit wird in Python begrenzt:
  Der Sollwert darf sich pro 20 ms um höchstens 7,5° ändern (375 °/s).
- **Auswertung (`--headless`):** umgefallen?, Strecke, Drift, Abweichung der
  Körperlage, höchstes Servo-Drehmoment.
- **Offene Steuerung:** Es gibt keine Rückkopplung. Der Trab weiß nicht, ob
  Nova kippt.

### 1.3 Robotermodell

- **Kein URDF, sondern MJCF** (MuJoCos XML), das `nova_model.py` zur Laufzeit
  aus den Maßen in `kinematics.py` erzeugt. Gut so: Die Maße stehen an einer
  Stelle, Kinematik und Simulation können nicht auseinanderlaufen.
- Einfache Formen statt STL: Rumpf als Quader, Beine als Kapseln, Füße als
  Kugeln (Radius 10 mm). Die Roboterteile stoßen nur mit dem Boden zusammen,
  nicht miteinander (keine Selbstkollision).
- Massen: Rumpf 1,0 kg, je Bein 0,22 kg, **zusammen 1,88 kg** (Platzhalter).
- Gelenke heißen wie in `servos.py`, Nullpunkt und Drehrichtung wie die
  Kinematik-Winkel. Gelenkbereiche (Kinematik-Winkel): Schulter −45..45°,
  Oberschenkel **−150..90°**, Knie 0..160°.
- Keine Sensoren im Modell (keine IMU, kein Abstandssensor).

### 1.4 Kinematik

- `leg_ik(x, y, z)` und `leg_fk()` für ein Bein (3 Gelenke: Schulter seitlich,
  Oberschenkel, Knie), geometrisch gelöst, wirft `ValueError` bei unerreichbaren
  Punkten.
- `body_ik(feet, BodyPose(height, roll, pitch, yaw))` für alle 4 Beine mit
  geneigtem Körper. Eine Rechnung dauert ca. 40 µs am Laptop.
- Maße sind Spot-Micro-Platzhalter: Hüfte 55 mm, Oberschenkel 107,5 mm,
  Unterschenkel 130 mm, Schulterachsen 186 × 78 mm.
- Im Stehen (170 mm Körperhöhe) liegen die Winkel bei Schulter 0°,
  Oberschenkel −50°, Knie 89°.
- Die Umrechnung Kinematik-Winkel → Servo-Winkel (Kalibrierung) fehlt noch.

### 1.5 Gangart

- Grundposen `STAND`, `SIT`, weiche Übergänge mit `blend()`.
- `Trot`: diagonaler Trab, Füße auf einer Zykloide, Standfüße rutschen nicht,
  Anlaufen und Anhalten ohne Sprung. Standard: 0,5 s Periode, 25 mm
  Schritthöhe, `duty` 0,6, 150 mm/s.
- **Nur geradeaus**: `update(dt, speed)` kennt keine Drehung und keinen
  Seitwärtsschritt. Für die Hindernisvermeidung muss Nova aber drehen können.
- Die Phase des Trabs ist privat (`_phase`), das Netz braucht sie aber als
  Eingang.

### 1.6 Servo-Ansteuerung

- `Servos.set_joint()` / `set_all()` über PCA9685 (`adafruit-circuitpython-servokit`,
  I2C), begrenzt **immer** auf die Gelenkgrenzen, NaN/None → Grundstellung,
  `inverted` für gespiegelte Servos. Simulationsmodus mit `NOVA_SIM=1`.
- `JOINT_ORDER` legt die Reihenfolge der 12 Werte fest (FL, FR, RL, RR, je
  Schulter/Ober/Unter).
- Pulsbreiten 500 bis 2500 µs passen schon zum Datenblatt. Grenzen
  (Servo-Winkel 60..120° Schulter, 30..150° sonst) und Neutralstellungen sind
  Platzhalter.
- Die Grenzen in `servos.py` (Servo-Winkel) und in `nova_model.py`
  (Kinematik-Winkel) sind **nicht miteinander verbunden**.

### 1.7 Abhängigkeiten

| Wo | Pakete |
|---|---|
| PC | `matplotlib`, `mujoco` (im venv `.venv`, keine requirements-Datei) |
| Pi | `adafruit-circuitpython-servokit` (nur am Pi, wird erst bei echtem Betrieb importiert) |

Fürs Netz kommen dazu: am PC `gymnasium`, `stable-baselines3`, `torch` (CPU
reicht), `tensorboard`, `onnx`, `onnxruntime`; am Pi `numpy`, `onnxruntime`,
`adafruit-circuitpython-mpu6050` (oder `smbus2`) und später `gpiozero`/`lgpio`
für den Ultraschall.

### 1.8 Hardware-Daten von euch

**Servos: Miuzei 25 kg, 180°, digital, Metallgetriebe**

| Wert | 4,8 V | 6,6 V (LiPo fast leer) | 7,4 V (LiPo Nennspannung) | 8,4 V (LiPo voll) |
|---|---|---|---|---|
| Blockiermoment | 23,5 kg·cm = 2,30 N·m | ≈ 2,50 N·m | ≈ 2,59 N·m | 27,5 kg·cm = 2,70 N·m |
| Tempo ohne Last | 0,15 s/60° = 400 °/s | ≈ 429 °/s | ≈ 443 °/s | 0,13 s/60° = 462 °/s |

Zwischenwerte linear geschätzt. Dazu: Strom 2,2 bis 3 A (wohl beim
Blockieren), Totzone 3 µs ≈ 0,27°, 180° bei 500 bis 2500 µs.

Was daraus folgt:

- **Die Servos dürfen direkt an den 2S-LiPo** (bis 8,4 V), ein
  Spannungswandler für die Servos ist nicht nötig.
- **Moment und Tempo hängen vom Ladezustand ab** (6,6 bis 8,4 V, also ca.
  ±4 %, dazu Spannungseinbruch unter Last). Das kommt in die Randomisierung.
- **Die Simulation rechnet bisher vorsichtig** (2,0 N·m, 375 °/s). Das ist
  gut, aber es fehlt die Kennlinie: Bei 300 °/s, wie im Trab, liefert ein
  Servo nur noch ca. **30 % seines Blockiermoments**. Weil die schnellen
  Bewegungen in der Schwungphase passieren (Bein in der Luft), geht das, muss
  aber in der Simulation stehen.
- **Der PCA9685 ist gröber als der Servo:** Bei 50 Hz hat er Stufen von
  4,9 µs ≈ 0,44°. Viele Digitalservos vertragen auch höhere PWM-Frequenzen,
  dann wären Stufen und Verzögerung kleiner. Das am Prüfstand testen, bevor
  man es einbaut.
- **Ohne Strom lassen sie sich frei drehen.** Beim ersten Signal springen sie
  mit voller Kraft auf den Sollwert. Das ist beim Einschalten wichtig (5.4).

**IMU: GY-521 mit MPU6050** (Beschleunigung + Drehrate, I2C, Adresse 0x68).
Reicht für das Netz, siehe 4.4.

**Ultraschall:** noch nicht ausgesucht, Vorschlag in 4.4.

---

## 2. Taugt die Simulation fürs Training?

### 2.1 Was ich gemessen habe

Alle Läufe: der Ablauf aus `demo_frames()` mit Trab 150 mm/s (geplant 900 mm).

| Versuch | Umgefallen | Strecke | Seitlich / gedreht | Max. Neigung | Max. Servo-Moment |
|---|---|---|---|---|---|
| Standard (1,88 kg, 2,0 N·m) | nein | 103 % | +37 mm / +3,9° | 1,9° | 1,73 N·m (86 %) |
| Verzögerung 20 ms bzw. 40 ms | nein | 103 % | gleich | 1,9° | gleich |
| Reibung **nur am Fuß** 0,08 | nein | 103 % | gleich | 1,9° | gleich |
| Reibung Fuß **und** Boden 0,3 | nein | 95 % | +18 mm / +1,6° | 1,9° | – |
| Reibung Fuß und Boden 0,15 | nein | 58 % | +68 mm / +0,9° | 5,8° | – |
| Kalibrierfehler ±3° je Gelenk (zufällig) | nein | **63 %** | **+350 mm / +43°** | 6,3° | 2,00 N·m (**100 %**) |
| Seitenstoß 5 N für 0,2 s | nein | 90 % | +19 mm / −21° | 5,6° | – |
| Seitenstoß 10 N für 0,2 s | **ja** | – | – | 56° | – |
| Masse 2,70 kg, Servo 2,0 N·m | nein | 101 % | −37 mm / −9° | 1,4° | 1,91 N·m (95 %) |
| Masse 2,70 kg, Servo 2,45 N·m (≈ eure Servos, Akku fast leer) | nein | 100 % | −30 mm / −8° | 1,4° | 1,89 N·m (77 %) |
| Masse 3,20 kg, Servo 2,45 N·m | nein | 99 % | −47 mm / −4° | 1,6° | 2,00 N·m (81 %) |

Tempo am Laptop (alter i5, 1 Kern): **ca. 2.800 Steuerschritte pro Sekunde**,
also 56-mal schneller als Echtzeit. Reine Physik ohne Python: 33.000
Physikschritte/s.

Was man daraus lernt:

1. **Die Fußreibung im Modell wirkt nicht.** MuJoCo nimmt bei einem Kontakt
   den größeren Reibwert der beiden Teile. Der Boden hat den Standardwert 1,0,
   deshalb ist `friction` am Fuß egal. Für Domain Randomization muss die
   Reibung am Boden (oder an beiden) verändert werden.
2. **Verzögerung ist beim jetzigen Trab egal**, weil er nicht auf Sensoren
   reagiert: Alles passiert nur später. Sobald ein Netz auf die IMU reagiert,
   wird Verzögerung entscheidend (ein Netz, das auf 0 ms trainiert wurde,
   reagiert am echten Hund zu spät und kann sich aufschaukeln).
3. **Kalibrierfehler sind das größte Problem des offenen Trabs.** ±3° pro
   Gelenk sind bei Hobby-Servos realistisch (Verzahnung am Servohorn: 25
   Zähne, also 14,4° pro Zahn, dazu Spiel im Getriebe). Nova dreht dann um 43°
   ab und die Servos laufen ans Limit. Genau hier kann ein Netz mit IMU helfen.
4. **Stöße fängt der offene Trab nicht ab.** 10 N für 0,2 s (ein kräftiger
   Schubs) werfen Nova um.
5. **Das Drehmoment ist knapp.** Ein echter Spot Micro mit 12 Servos
   (je ca. 60 bis 70 g), Pi, LiPo und Powerbank wiegt eher 2,5 bis 3 kg.
   Mit euren Servos reicht es in der Simulation (77 % bei 2,7 kg), aber das
   Datenblatt-Moment ist das **Blockiermoment**: Dauerhaft sollte man nur mit
   ca. der Hälfte rechnen. Masse und Servodaten müssen also stimmen, sonst
   lernt das Netz Bewegungen, die die echten Servos nicht schaffen.

### 2.2 Bewertung Punkt für Punkt

| Punkt | Stand | Realistisch? | Was fürs Training nötig ist |
|---|---|---|---|
| Physik-Engine | MuJoCo, 2 ms, Kontakte | ja | passt, MuJoCo ist Standard für Lauf-RL |
| Geometrie | Spot-Micro-Platzhalter, einfache Formen | ungefähr | Maße aus den STL-Dateien; einfache Formen reichen |
| Massen | 1,88 kg, gleichmäßig | eher zu leicht | Teile wiegen, Schwerpunkt mit Akku und Powerbank richtig setzen; bis dahin mit ca. 2,6 kg rechnen und stark randomisieren |
| Gelenkbereiche | Oberschenkel −150..90° = 240° | **nein**, Servo kann 180° | Bereiche aus der Kalibrierung, eine gemeinsame Quelle für `servos.py` und `nova_model.py` |
| Servo-Moment | konstant 2,0 N·m | vorsichtig, ohne Kennlinie | 2,5 bis 2,7 N·m je nach Akku, **Moment-Drehzahl-Kennlinie** (bei 300 °/s nur ca. 30 %), Spannungseinbruch |
| Servo-Tempo | Sollwert-Rampe 375 °/s | etwas zu langsam | 430 bis 460 °/s ohne Last, am Prüfstand messen |
| Servo-Regler | P-Regler ohne Dämpfung | grob | Dämpfung, Totzone 0,27°, PCA9685-Stufen 0,44°, Spiel; aus Messungen anpassen |
| Verzögerung | keine | **nein** | 20 ms PWM-Periode + Servo-intern + Pi-Schleife: 5 bis 40 ms randomisieren |
| Reibung | wirkt nicht (siehe oben), Boden 1,0 | **Fehler** | am Boden setzen, 0,4 bis 1,2 randomisieren |
| Füße | starre Kugel 10 mm | ungefähr | Gummi ist nachgiebig: Kontakt-Weichheit (`solref`) randomisieren |
| IMU | fehlt | – | **Pflicht:** `site` im Rumpf mit `gyro` und `accelerometer`, Rauschen dazu, **derselbe Lagefilter wie am Pi** |
| Ultraschall | fehlt | – | `rangefinder`-Sensoren (mehrere Strahlen für den Kegel) und Hindernisse |
| Selbstkollision | aus | ok für Stufe 1 | **für Stufe 2 nötig**, sonst lernt das Netz, Beine durcheinander zu schwingen |
| Gelände | nur flach | – | später kleine Unebenheiten und leichte Schräge |
| 3D-Druck biegt sich | nicht modelliert | – | über Randomisierung (Kalibrierfehler, Spiel) abdecken |
| Rückkopplung | keine | – | kommt mit IMU und Netz |
| Drehen | Trab nur geradeaus | – | `Trot.update()` um Drehrate erweitern |

### 2.3 Fazit

Die Grundlage ist gut: echte Physik, schnelle Simulation, Gelenke und Winkel
passen zur Kinematik, Servos sind schon kraft- und tempobegrenzt. Nach den
Ergänzungen aus Phase 1 (Abschnitt 6) reicht sie für Stufe 1. Für Stufe 2
braucht es zusätzlich Selbstkollision und ein Servo-Modell, das mit den
gemessenen Servos abgeglichen ist.

---

## 3. Welcher Weg?

### 3.1 Die Möglichkeiten

| | Klassisch + Regler | **Stufe 1: Netz verbessert den Trab** | **Stufe 2: Netz läuft selbst, mit Vorbild** | Netz läuft selbst, von null | Netz für Hindernisse |
|---|---|---|---|---|---|
| Was macht das Netz? | kein Netz, IMU-Regler korrigiert Roll/Pitch | korrigiert die Fußpunkte des Trabs | gibt alle 12 Gelenkwinkel aus; lernt erst den Trab nachzumachen, dann besser zu werden | gibt alle 12 Gelenkwinkel aus, lernt alles selbst | wählt Tempo und Drehrichtung aus dem Abstand |
| Trainingsdauer (PC, CPU) | – | ca. 1 bis 3 h je Lauf | ca. 3 bis 10 h je Lauf | 10 bis 50 h je Lauf, viele Läufe | Minuten |
| Risiko Sim-to-Real | gering | mittel | mittel bis hoch | **hoch** | gering |
| Sicherheit | gut | gut: Korrektur begrenzt, Netz = 0 ist der Trab | Netz kann jede Pose befehlen, Begrenzung und Überwachung nötig | wie links | gut |
| Erfüllt "Netz läuft selbst" | nein | teilweise | **ja** | ja | nein |
| Wert für die Arbeit | gering | hoch: messbar besser als klassisch | **sehr hoch**: drei Verfahren im Vergleich | sehr hoch, wenn es klappt | mittel (ein Abstandswert ist wenig Information) |

### 3.2 Empfehlung: Muss, Soll, Kann

Du glaubst, das Netz soll **selbst laufen und den Gang verbessern**. Das deckt
der folgende Plan ab, ohne dass alles an einem riskanten Schritt hängt:

| Ziel | Inhalt |
|---|---|
| **Muss** | realistische Simulation (Servo-Modell, IMU); **Stufe 1** (Netz verbessert den Trab) in der Simulation messbar besser als klassisch; Netz läuft in Echtzeit am Pi; klassischer Trab am echten Hund |
| **Soll** | **Stufe 2** (Netz läuft selbst) in der Simulation; Stufe 1 und 2 am echten Hund, Vergleich klassisch / Stufe 1 / Stufe 2 |
| **Kann** | Drehen auf Befehl, Hindernisvermeidung mit Ultraschall (Regeln, dann Netz), unebener Boden |

Begründung:

- **Beide Stufen nutzen denselben Unterbau** (Simulation, Umgebung,
  Beobachtungen, Belohnung, Export, Pi-Schleife). Stufe 1 ist also keine
  Extra-Arbeit, sondern der erste Test dieses Unterbaus.
- **Stufe 1 bringt schnell ein sicheres Ergebnis.** Das Netz muss nicht das
  Laufen von null lernen, nur wie man den getesteten Trab verbessert. Das
  dauert Stunden statt Tage. Am echten Hund kann man es mit einem Regler
  (0 % → 25 % → 100 %) stufenweise zuschalten.
- **Stufe 1 ist der Lehrer für Stufe 2.** Das Netz der Stufe 2 lernt zuerst,
  den Trab samt den Korrekturen aus Stufe 1 nachzumachen. Dann läuft es schon
  nach Minuten Training selbst, und PPO muss es nur noch verbessern. Von null
  zu lernen dauert viel länger und lernt oft seltsame Gangarten (Hüpfen,
  Zittern), die echte Servos nicht schaffen.
- **Eure Servos melden ihren Winkel nicht zurück** und haben Spiel und
  Verzögerung. Je mehr das Netz selbst macht, desto empfindlicher ist es
  dafür. Mit der Reihenfolge Stufe 1 → Stufe 2 habt ihr immer einen
  funktionierenden Stand.
- **Gut auswertbar:** Drei Verfahren im Vergleich, bei Stößen,
  Kalibrierfehlern, glattem Boden, schwerer Last. Die Messungen in 2.1 sind
  die Ausgangswerte: Der klassische Trab fällt bei 10 N um und dreht bei ±3°
  Kalibrierfehler um 43° ab.
- **Das gibt es so schon.** *spot_mini_mini* (Rahme u. a., 2020) hat Stufe 1
  mit einem Spot-Micro-Klon gemacht und auf echte Hardware übertragen
  (allgemein heißt die Idee *Policies Modulating Trajectory Generators*). Das
  Nachahmen mit anschließendem RL (Stufe 2) kommt aus *DeepMimic* bzw.
  *Learning Agile Robotic Locomotion Skills by Imitating Animals* (Peng u. a.,
  2018/2020), dort auch auf einem echten Vierbeiner.

**Mit dem Betreuer abklären.** Vorschlag für die Frage:

> Wir planen das neuronale Netz in zwei Stufen. Stufe 1: Ein Netz verbessert
> unseren klassischen Trab mit der IMU (Gleichgewicht, Geradeauslauf, Stöße).
> Stufe 2: Ein Netz steuert alle 12 Gelenke selbst. Es lernt zuerst, den Trab
> nachzumachen, und wird dann mit Reinforcement Learning in der Simulation
> verbessert. Der klassische Trab bleibt als Rückfallebene. Passt es, wenn
> Stufe 1 das Muss-Ziel und Stufe 2 das Soll-Ziel ist?

Falls der Betreuer sagt, dass Stufe 2 Pflicht ist: Der Plan bleibt gleich, nur
die Einteilung in Muss und Soll ändert sich. Dann sollte der Hund noch früher
fertig sein.

---

## 4. Das Netz: Beobachtungen, Aktionen, Belohnung

Grundregel: **Das Netz darf nur sehen, was es auch am echten Hund gibt.**
Eure Servos melden keinen Winkel zurück, also gibt es keine gemessenen
Gelenkwinkel, sondern nur die befohlenen. Die echte Geschwindigkeit des Hundes
kennt man am Pi auch nicht. Solche Werte dürfen in die Belohnung (die gibt es nur
in der Simulation), aber nicht in die Beobachtung.

Steuertakt in beiden Stufen 50 Hz, wie der Trab und die PWM des PCA9685.

### 4.1 Stufe 1: Netz verbessert den Trab

**Beobachtungen (Eingänge), 34 Werte**

| Werte | Anzahl | Herkunft am Pi | Skalierung (Vorschlag) |
|---|---|---|---|
| Schwerkraft-Richtung im Rumpf | 3 | MPU6050 + Lagefilter | schon −1..1 |
| Winkelgeschwindigkeit Roll/Pitch/Yaw | 3 | MPU6050 (Gyroskop) | ÷ 4 rad/s |
| Befehl: Tempo vorwärts, Drehrate | 2 | Bediener bzw. Hindernis-Schicht | ÷ Maximalwert |
| Phase des Trabs als sin, cos | 2 | `gait.Trot` | schon −1..1 |
| letzte Aktion des Netzes | 12 | gemerkt | schon −1..1 |
| zuletzt befohlene Gelenkwinkel (Kinematik) | 12 | gemerkt | (Winkel − Stehwinkel) ÷ 45° |

Optional: die IMU-Werte der letzten 2 Schritte zusätzlich (+12 Werte). Das hilft
dem Netz, Verzögerung auszugleichen. Feste Skalierung statt `VecNormalize`
macht den Export einfacher (keine Statistik-Datei, die am Pi fehlen kann).

**Aktionen (Ausgänge), 12 Werte**

Pro Fuß Δx, Δy, Δz im Bereich −1..1, skaliert auf ±20 mm (x), ±10 mm (y),
±15 mm (z). Ablauf in jedem Schritt:

```
Trab-Fußpunkte  +  Netz-Korrektur (begrenzt, leicht geglättet)
        → body_ik()            (bei ValueError: Korrektur dieses Beins = 0)
        → Gelenkgrenzen        (dieselben Grenzen wie am echten Hund)
        → Servo-Modell (Sim) bzw. servos.set_all() (Pi)
```

Alternative: Korrektur direkt auf die Gelenkwinkel (±10°) nach der IK. Das ist
einfacher (keine IK-Fehler), aber Fußpunkte sind anschaulicher, übertragen sich
besser und sind in der Arbeit leichter zu erklären.

**Belohnung pro Schritt (Startwerte, werden beim Training angepasst)**

| Term | Formel | Gewicht | Wozu |
|---|---|---|---|
| Tempo halten | exp(−(vₓ − vₓ,soll)² / 0,05²) | +1,0 | soll das befohlene Tempo laufen |
| Drehrate halten | exp(−(ω_z − ω_z,soll)² / 0,3²) | +0,5 | geradeaus heißt ω_z = 0, also kein Abdriften |
| seitliches Rutschen | v_y² | −2,0 | keine Drift |
| Neigung | g_x² + g_y² (Schwerkraft im Rumpf) | −2,0 | Rumpf gerade halten |
| Wackeln | ω_x² + ω_y² | −0,05 | ruhiger Lauf |
| Hüpfen | v_z² | −1,0 | ruhiger Lauf |
| Größe der Korrektur | ‖a‖² / 12 | −0,3 | nur korrigieren, wenn es etwas bringt |
| Ruckeln | ‖aₜ − aₜ₋₁‖² / 12 | −0,5 | wichtig für Servos und Getriebe |
| Servo-Belastung | mittel((τ / τ_max)²) | −0,2 | Servos und Akku schonen |
| Servo am Anschlag | Anteil Servos mit \|τ\| > 0,95 τ_max | −0,5 | nichts lernen, was nur mit voller Kraft geht |
| Umfallen | einmalig beim Abbruch | −10 | |

Abbruch der Episode: Rumpf unter 80 mm, Neigung über 40° oder Rumpf berührt
den Boden. Episode 10 s (500 Schritte), Befehl wechselt alle 2 bis 4 s
(Tempo 0 bis 200 mm/s, später Drehrate ±0,5 rad/s).

**Netz:** kleines MLP, 2 Schichten mit je 128 Neuronen (`net_arch=[128, 128]`),
für Actor und Critic getrennt. Das sind ca. 25.000 Rechenoperationen pro Schritt,
am Pi 5 weit unter 1 ms.

**PPO-Startwerte (SB3):** so viele Umgebungen parallel wie CPU-Kerne,
`n_steps=1024`, `batch_size=2048`, `n_epochs=10`, `learning_rate=3e-4`,
`gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `log_std_init=-1` (kleine
Zufallsausschläge am Anfang, damit der Trab nicht gleich kaputt korrigiert
wird). Ziel: ca. 10 bis 20 Millionen Schritte.

### 4.2 Stufe 2: Netz läuft selbst (mit Vorbild)

Gleiche Umgebung, nur ein anderer Modus (`mode="full"` statt `"residual"`).

**Beobachtungen, 34 Werte:** wie Stufe 1, aber die "zuletzt befohlenen
Gelenkwinkel" fallen weg (die sind jetzt gleich der letzten Aktion), dafür
kommen die IMU-Werte der letzten 2 Schritte dazu (+12). Die Phase bleibt als
Eingang: Sie ist nur ein Taktgeber wie ein Metronom, die Bewegung selbst kommt
komplett aus dem Netz.

**Aktionen, 12 Werte:** Gelenkwinkel relativ zur Stehpose, −1..1 skaliert auf
±30° (Schulter ±15°). Dann Gelenkgrenzen, Begrenzung der Änderung pro Schritt
(z. B. höchstens 8° pro 20 ms), Servo-Modell bzw. `servos.set_all()`.

**Ablauf des Trainings**

1. **Daten sammeln:** Der Lehrer (Trab + Netz aus Stufe 1) läuft in der
   Simulation mit Randomisierung. Gespeichert werden Beobachtung und
   befohlene Gelenkwinkel, ca. 1 Million Paare (einige Minuten).
2. **Behavior Cloning:** Ein Netz mit derselben Form wie der PPO-Actor lernt
   die Gelenkwinkel aus den Beobachtungen nachzumachen (überwachtes Lernen,
   mittlerer quadratischer Fehler, Minuten am PC). **Schon danach läuft Nova
   in der Simulation mit dem Netz allein.** Das ist ein guter erster
   Meilenstein.
3. **PPO-Feinschliff:** Der Actor startet mit den Gewichten aus Schritt 2.
   Der Critic wird zuerst einige Updates allein trainiert (Actor eingefroren),
   sonst macht der anfangs unsinnige Critic den Actor kaputt.
   Belohnung wie Stufe 1 plus:

| Term | Formel | Gewicht | Wozu |
|---|---|---|---|
| Nachahmung | exp(−‖q − q_ref‖² / 0,3²), q_ref = Trab bei gleicher Phase | +1,0 → +0,2 | Vorbild, wird im Lauf des Trainings ausgeblendet, damit das Netz besser werden darf als der Trab |
| Füße rutschen | Fußgeschwindigkeit bei Bodenkontakt (nur Sim) | −0,5 | sauberes Aufsetzen |
| Fußhöhe | Schwungfuß unter 15 mm | −0,5 | nicht schlurfen |
| Ruckeln | ‖aₜ − aₜ₋₁‖² / 12 | −1,0 (stärker als bei Stufe 1) | Servos schonen |

Für Stufe 2 wird die Selbstkollision im Modell eingeschaltet. Geschätzt sind
30 bis 100 Millionen Schritte nötig, also eine Nacht pro Lauf am PC.

### 4.3 Kann-Ziel: Hindernisse mit dem Ultraschall

Sitzt über Stufe 1 oder 2 und gibt nur den Befehl (Tempo, Drehrate) vor, mit
ca. 10 Hz.

1. **Zuerst Regeln** (`src/obstacle.py`): frei → laufen, unter 40 cm → langsamer,
   unter 25 cm → anhalten und drehen, bis frei. Das ist die Vergleichsbasis.
2. **Optional ein Netz:** Eingänge die letzten 3 Abstände (÷ 2 m), aktueller
   Befehl; Ausgänge Tempo (0..max) und Drehrate (±max); Belohnung + Strecke
   ohne Zusammenstoß, −10 bei Abstand unter 15 cm, kleiner Abzug fürs Drehen.
   Training in einer eigenen 2D-Simulation (Punkt mit Richtung in zufälligen
   Räumen mit Hindernissen, Ultraschall als Kegel mit Rauschen). Braucht kein
   MuJoCo und ist in Minuten fertig. Sinnvoll erst mit 2 Sensoren (siehe 4.4),
   mit einem einzigen Abstandswert ist eine Regel fast genauso gut.

Voraussetzung: Der Trab bzw. das Netz muss drehen können (Erweiterung von
`gait.Trot`, Drehrate als Befehl).

### 4.4 Sensoren und Hardware

**IMU: euer GY-521 (MPU6050) passt.** Das Netz braucht nur die Richtung der
Schwerkraft im Rumpf (Roll, Pitch) und die Drehraten. Yaw braucht es nicht,
die Drift des MPU6050 beim Yaw ist also egal.

- **Anschluss:** VCC an 3,3 V (Pin 1), GND, SDA an GPIO 2 (Pin 3), SCL an
  GPIO 3 (Pin 5). Gleicher I2C-Bus wie der PCA9685 (0x40), der MPU6050 hat
  0x68, kein Konflikt. `i2cdetect -y 1` zeigt 0x40, 0x68 und 0x70 (Sammeladresse
  des PCA9685).
- **Klone prüfen:** Viele günstige GY-521 haben einen Nachbau-Chip. Das
  Register `WHO_AM_I` (0x75) sollte 0x68 liefern. Wenn nicht, verweigern
  manche Bibliotheken den Dienst, dann direkt mit `smbus2` auslesen.
- **Einstellungen:** Gyroskop ±500 °/s, Beschleunigung ±4 g, digitaler
  Tiefpass (`DLPF`) ca. 20 bis 44 Hz gegen die Vibrationen der Servos.
- **Lagefilter:** Komplementärfilter (Roll/Pitch aus der Beschleunigung, mit
  dem Gyroskop geglättet), ca. 15 Zeilen Python, gut in der Arbeit zu
  erklären. Beim Start 2 s still halten und den Nullpunkt des Gyroskops
  mitteln. Einmal den Einbauwinkel messen (Hund steht gerade) und abziehen.
- **Gleicher Filter in der Simulation:** MuJoCo liefert mit `accelerometer`
  und `gyro` genau die Rohwerte, die auch der MPU6050 misst. Darauf kommen
  Rauschen und Nullpunktfehler, dann läuft **derselbe Filtercode** aus
  `src/imu.py`. So kennt das Netz die Schwächen des Filters schon aus dem
  Training (z. B. dass die Beschleunigung beim Aufsetzen der Füße stört).
- **Einbau:** fest und möglichst nahe der Rumpfmitte, mit dünnem
  Schaumstoff-Klebeband gegen Vibrationen. Achsen am besten so ausrichten
  wie die Körper-Koordinaten (x vorne, y links, z oben), sonst eine
  Umrechnungstabelle im Code.
- Bibliothek: `adafruit-circuitpython-mpu6050` (passt zu Blinka, das die
  Servo-Bibliothek schon mitbringt) oder `smbus2` direkt.

**Ultraschall (Kann-Ziel, eilt nicht):**

| Möglichkeit | Vorteil | Nachteil |
|---|---|---|
| **RCWL-1601 oder HC-SR04P** an 3,3 V (Empfehlung) | billig, direkt am Pi 5 ohne Spannungsteiler, `gpiozero.DistanceSensor` | Kegel 15 bis 30°, weiche Flächen schlucken den Schall |
| HC-SR04 (5-V-Version) | überall erhältlich | **Echo-Pin hat 5 V, der Pi 5 verträgt nur 3,3 V:** Spannungsteiler nötig (z. B. 1 kΩ / 2 kΩ) |
| VL53L1X (Laser-Abstand) | I2C (0x29), klein, genau | Probleme mit Glas und sehr dunklen Flächen |

Mit **zwei Sensoren** (ca. ±20° nach links und rechts gedreht) kann Nova
entscheiden, in welche Richtung es ausweichen soll. Dann lohnt sich auch ein
Netz. Die beiden Sensoren abwechselnd auslösen, sonst stören sie sich.

**Weitere Hardware**

| Was | Wofür | Pflicht? | Hinweise |
|---|---|---|---|
| **Stromverteilung Servos** | bis 3 A pro Servo, bei 12 Servos Spitzen weit über 10 A | **ja** | Servostrom **nicht** über die Schraubklemme und Leiterbahnen des PCA9685 führen, sondern über eine eigene Verteilung mit dicken Kabeln und Sicherung. LiPo mit ausreichender C-Rate. Masse von LiPo und Pi verbinden. |
| **Not-Aus** | Servostrom hart abschalten | **ja** | Schalter in der Servo-Versorgung. Zusätzlich den OE-Pin des PCA9685 an einen GPIO: Damit schaltet die Software alle Signale auf einmal ab. **Wichtig:** Stürzt der Pi ab, gibt der PCA9685 einfach das letzte Signal weiter aus. |
| Akkuspannung (INA226 oder ADS1115) | LiPo nicht tiefentladen (unter 3,3 V pro Zelle = 6,6 V), Spannung für die Auswertung | empfohlen | INA226 misst auch den Strom (Energie pro Meter für die Arbeit). **INA226 auf Adresse 0x41 stellen**, 0x40 hat schon der PCA9685. ADS1115 hat 0x48. |
| Gelenkwinkel-Rückmeldung | echte Winkel für das Netz | nein | Ginge nur mit Umbau (Draht am Servo-Potentiometer + ADC). Fürs Netz nicht nötig. Für die Servo-Messung reicht es, **einen** Test-Servo so umzubauen. |
| Fußkontakt-Sensoren | – | nein | Bei festen Gummifüßen unnötig. |

---

## 5. Sim-to-Real

### 5.1 Servo-Modell und Messungen (Systemidentifikation)

Sobald die Servos da sind, und noch **bevor** der Hund gebaut ist. Am besten
alles bei 7,4 V und einmal bei 8,4 V.

| Messung | Wie | Ergebnis für die Simulation |
|---|---|---|
| Tempo ohne Last | Sprung um 90° befehlen, mit Handy-Zeitlupe (240 fps) und Winkelskala filmen | ω_max (Datenblatt: 430 bis 460 °/s) |
| Tempo mit Last | wie oben, mit Gewicht am Hebel | Kennlinie Moment ↔ Drehzahl |
| Blockiermoment | Hebel am Servohorn (z. B. 10 cm), Gewichte anhängen, bis er nachgibt | τ_stall (Datenblatt: 2,5 bis 2,7 N·m) |
| Verzögerung | LED gleichzeitig mit dem Befehl einschalten, filmen, Bilder bis zur ersten Bewegung zählen | Totzeit |
| Spiel | Horn mit Laserpointer, von Hand hin und her drücken, Ausschlag an der Wand messen | Spiel in ° |
| PWM-Frequenz | 50, 100, 200 Hz am PCA9685 probieren, ob der Servo sauber folgt | kleinere Stufen und Verzögerung? |
| Sprungantwort | Test-Servo mit Draht am Potentiometer + ADS1115, Winkel über der Zeit | Regler-Steifigkeit, Dämpfung |
| Strom | Strommessgerät bzw. INA226 bei Last | Akku, Kabel, Spannungseinbruch |
| Massen, Schwerpunkt | jedes Teil wiegen, Rumpf mit Akku auf einer Kante ausbalancieren | Massen im Modell |

Daraus wird ein Servo-Modell für `sim.py`, das jeden Physikschritt (2 ms)
rechnet:

```
Sollwert auf PCA9685-Stufen runden (0,44°), verzögern (Totzeit), Totzone (0,27°)
τ = Kp · (Soll − Ist) − Kd · Drehzahl
τ_max(Drehzahl) = τ_stall · (1 − |Drehzahl| / ω_max)     # bei voller Drehzahl kein Moment
τ auf ±τ_max begrenzen
```

Technisch: MuJoCo-`motor`-Aktuatoren statt `position`, das Moment rechnet
Python. Das ist langsamer als jetzt (geschätzt 1.500 statt 2.800 Schritte/s),
aber nachvollziehbar und mit den Messungen vergleichbar. Bis die Messungen da
sind, mit den Datenblattwerten rechnen und breit randomisieren.

### 5.2 Domain Randomization

Bei jedem Episoden-Start neu gewürfelt. Erst ohne trainieren, dann
schrittweise einschalten (Curriculum), sonst lernt das Netz am Anfang gar nichts.

| Was | Bereich (Start) | Warum |
|---|---|---|
| Rumpfmasse | ±20 % | Akku, Powerbank, Kabel |
| Schwerpunkt Rumpf | ±2 cm in x und y | Lage von Akku und Powerbank |
| Beinmassen | ±15 % | |
| Reibung **Boden** | 0,4 bis 1,2 | Laminat, Fliesen, Teppich; am Boden setzen, siehe 2.1 |
| Kontakt-Weichheit Fuß | `solref` 0,01 bis 0,04 | Gummifüße |
| Servo-Moment | 2,0 bis 2,7 N·m Blockiermoment | Akku 6,6 bis 8,4 V, Spannungseinbruch, Wärme, Streuung |
| Servo-Tempo | 380 bis 460 °/s | Akku, Streuung |
| Regler Kp, Kd | ±30 % | |
| Totzeit | 5 bis 40 ms (in 2-ms-Schritten) | PWM 20 ms + Servo + Pi |
| Kalibrierfehler | ±3° pro Gelenk, fest pro Episode | Servohorn-Verzahnung, Biegung |
| Spiel/Totzone | 0 bis 2° | Getriebe |
| IMU | Rauschen, Nullpunktfehler Gyroskop, Einbauwinkel ±2°, 0 bis 10 ms Verzögerung | MPU6050 |
| Steuertakt | 50 Hz ± 2 ms | Python am Pi |
| Stöße | alle 3 bis 6 s zufällig 0 bis 15 N für 0,1 bis 0,2 s | Robustheit, auswertbar |
| Boden | erst flach, später ±1 cm Unebenheiten, ±5° Schräge | |

### 5.3 Training am PC

- **Dein PC mit RX 6700 XT reicht, gerechnet wird auf der CPU.** Bei SB3 mit
  so kleinen Netzen bringt eine Grafikkarte nichts, die Zeit geht in die
  Physik (PyTorch unter Windows nutzt AMD-Grafikkarten ohnehin kaum).
  Entscheidend ist die Zahl der CPU-Kerne: Es laufen so viele Umgebungen
  parallel, wie Kerne da sind (`SubprocVecEnv`).
- **Die NVIDIA-PCs der anderen** sind erst interessant, falls Stufe 2 auf der
  CPU zu langsam ist. Dann ginge GPU-Training mit MJX/MuJoCo Playground, aber
  nur unter Linux oder WSL2 und mit eigenem Aufwand. Einfacher ist es,
  **auf mehreren PCs gleichzeitig verschiedene Versuche** laufen zu lassen
  (andere Belohnungsgewichte, andere Zufallszahlen). Das beschleunigt das
  Ausprobieren am meisten.
- **Geschätzte Dauer** (8 Kerne): 3.000 bis 8.000 Schritte/s, mit
  Servo-Modell eher die Hälfte. Stufe 1 (10 bis 20 Mio. Schritte) dauert 1 bis 3 h,
  Stufe 2 (30 bis 100 Mio.) 3 bis 10 h, also über Nacht. Der Server-Laptop
  ist dafür zu schwach (4 alte Kerne, und er betreibt andere Dienste).
- **Windows:** `SubprocVecEnv` braucht `if __name__ == "__main__":` im
  Trainingsskript. `torch` ohne CUDA spart Platz:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Aufzeichnen:** TensorBoard in `runs/`, Zwischenstände in `checkpoints/`
  (beides schon in `.gitignore`). Fertige Netze nach `model/policies/`
  (werden committet).
- **Immer gegen den klassischen Trab auswerten**, mit derselben Randomisierung
  und denselben Zufallszahlen.

### 5.4 Export und Pi

1. **Export (`model/export_onnx.py`):** Aus dem SB3-Modell nur den Actor nehmen
   (`policy.mlp_extractor` + `policy.action_net`, deterministisch) und mit
   `torch.onnx.export` speichern. Dazu eine `.json` mit allem, was die
   Beobachtung braucht (Modus, Skalierungen, Reihenfolge, Aktionsgrenzen,
   Trab-Parameter, Versionsnummer). Danach prüfen: PyTorch und
   `onnxruntime` geben für 1.000 zufällige Beobachtungen dasselbe aus
   (Abweichung unter 1e-5).
2. **Am Pi:** `onnxruntime` im venv `~/nova/.venv` (gibt es fertig für
   ARM64). PyTorch braucht der Pi nicht.
3. **Ein Code für Sim und Pi:** Beobachtung bauen, Netz rechnen, Ausgabe
   begrenzen und glätten steht **einmal** in `src/policy.py`, der Lagefilter
   einmal in `src/imu.py`. Die Trainingsumgebung und die Pi-Schleife benutzen
   genau diese Dateien. Häufigster Sim-to-Real-Fehler sonst: eine andere
   Reihenfolge oder Skalierung am Pi.
4. **Hauptschleife (`src/controller.py`) mit 50 Hz:** MPU6050 lesen → Filter →
   Beobachtung → Netz → (Stufe 1: Trab + Korrektur → IK) → Grenzen →
   `servos.set_all()`. Zeiten mitloggen (CSV), damit man Jitter und
   Verzögerung am echten Hund sieht.
5. **Einschalten:** Ohne Strom lassen sich die Servos frei drehen, beim ersten
   Signal springen sie mit voller Kraft. Deshalb: Hund vor dem Einschalten in
   die Liegepose legen, als Erstes genau diese Pose senden, dann
   Servostrom an und langsam aufstehen.
6. **Sicherheit in der Schleife:**
   - Netz-Anteil `k` (0 bis 1) als Regler, startet bei 0. In Stufe 2 heißt
     das: weich zwischen Trab und Netz überblenden.
   - Neigung über 30° → anhalten, hinsetzen, Servos entspannen.
   - Schleife zu spät (mehr als 3 Takte), IMU liefert nichts, NaN aus dem Netz →
     klassischer Trab bzw. Stehen.
   - Akku unter 6,6 V → hinsetzen, entspannen.
   - Alle Winkel laufen weiter über `servos.py` und werden dort begrenzt.

### 5.5 Erste Versuche am echten Hund (Reihenfolge)

1. Jeder Servo einzeln, Richtung und Nullpunkt prüfen (Kalibrierung).
2. Hund aufgebockt, Füße in der Luft: Stehen, Sitzen, Trab ohne Netz.
3. Am Boden mit Sicherungsleine: Stehen, Trab ohne Netz. Logs aufzeichnen.
4. **Dieselben Befehle in der Simulation abspielen und mit den Logs
   vergleichen** (IMU-Verlauf, Strecke). Abweichungen → Modell anpassen → neu
   trainieren. Das ist zugleich ein schönes Kapitel für die Arbeit.
5. Stufe 1 mit `k` = 0,25, dann 0,5, dann 1, immer gegen den Trab ohne Netz
   vergleichen.
6. Stufe 2 erst aufgebockt, dann am Boden mit Leine, Überblendung vom Trab ins
   Netz.

### 5.6 Vorher am PC: der ganze Pi-Code gegen die Simulation

`controller.py` kann am PC statt gegen echte Servos und IMU gegen die
MuJoCo-Simulation laufen (Software-in-the-Loop). So ist die komplette
Pi-Software getestet, bevor der Hund gebaut ist. Das passt zur Regel aus
`CLAUDE.md`, dass alles ohne Hardware mit `NOVA_SIM=1` am PC laufen muss.

---

## 6. Zeitplan bis April 2027

Ca. 20 Wochen. Software (links) und Hardware (rechts) laufen parallel. Die
Arbeit nebenbei mitschreiben: Jede Phase liefert ein Kapitel.

| Zeitraum | KW | Software: Simulation und Netz | Hardware (parallel) |
|---|---|---|---|
| 16.11. bis 6.12.2026 | 47 bis 49 | **Phase 1:** Simulation realistischer | Betreuer fragen, Teile bestellen, Servos am Prüfstand messen, Druck starten |
| 7.12. bis 20.12. | 50 bis 51 | **Phase 2:** Trainingsumgebung | MPU6050 am Pi in Betrieb nehmen, drucken |
| 21.12.2026 bis 10.1.2027 (Ferien) | 52 bis 1 | **Phase 3:** Training Stufe 1, Export | drucken, zusammenbauen |
| 11.1. bis 31.1. | 2 bis 4 | **Phase 4:** Stufe 2 in der Simulation; Pi-Software gegen die Simulation | **Hund fertig gebaut**, kalibriert, Trab ohne Netz |
| 1.2. bis 28.2. | 5 bis 8 | Simulation mit echten Logs nachschärfen, neu trainieren | **Phase 5:** echter Hund: klassisch → Stufe 1 → Stufe 2 |
| 1.3. bis 21.3. | 9 bis 11 | **Phase 6:** Kann-Ziele (Drehen, Ultraschall) | Messungen für die Arbeit |
| 22.3. bis Abgabe | 12 bis 14 | **Phase 7:** Puffer, Auswertung, Arbeit fertigschreiben | |

**Kritischer Pfad:** Der Hund muss bis Ende Jänner stehen. Wird er später
fertig, fällt zuerst Stufe 2 am echten Hund weg (sie bleibt dann ein
Simulationsergebnis), dann die Kann-Ziele.

### Die Phasen im Detail

Jede Phase endet mit etwas Prüfbarem.

**Phase 1: Simulation realistischer machen**
- `nova_model.py`: Reibung am Boden setzen, IMU-`site` mit `gyro` und
  `accelerometer`, `rangefinder` vorne, Servodaten aus dem Datenblatt, Massen
  geschätzt (ca. 2,6 kg) mit Akku und Powerbank an der richtigen Stelle,
  Selbstkollision als Schalter.
- Gelenkgrenzen an einer Stelle festlegen (Kalibrierung in `servos.py`:
  Kinematik-Winkel ↔ Servo-Winkel), `nova_model.py` übernimmt sie.
- `sim.py`: Servo-Modell (5.1) mit Totzeit, Kennlinie, Totzone, PCA9685-Stufen.
- `src/imu.py`: Komplementärfilter, in der Simulation auf die MuJoCo-Rohwerte.
- `gait.py`: Drehrate im Trab, Phase öffentlich.
- `model/evaluate.py`: Kennzahlen des klassischen Trabs unter Randomisierung
  (wie Abschnitt 2.1, aber systematisch, mit festen Zufallszahlen).
- Ergebnis: Der klassische Trab läuft noch, die Ausgangswerte für den
  Vergleich stehen fest.
- **Stand 15.11.2026: erledigt** (lokal, noch nicht am echten Hund geprüft).
  Ausgangswerte in `model/results/trab-phase1.json` (`python model/evaluate.py`).
  Standard: 90 % Strecke, +2 mm / −0,3°, 1,4° Neigung, 61 % Moment.
  Kalibrierfehler ±3°: 76 %, +392 mm / +41°. Spiel 2°: 63 %. Stoß 15 N wird
  mit dem schwereren Modell überstanden, ab 20 N fällt Nova um. Randomisiert
  (20 Durchläufe): im Mittel 73 % Strecke, 190 mm Drift, 29° Drehung, mit
  Stößen fällt 1 von 20 um. Der IMU-Filter liegt nominal bis 1,9° daneben,
  unter starker Randomisierung (Hund federt) bis ca. 10°.
  **Achtung Phase 3:** Das Abnahme-Kriterium „15-N-Stoß“ unterscheidet nicht
  mehr, weil schon der Trab ihn übersteht. Vorschlag: 20 N.

**Phase 2: Trainingsumgebung**
- `src/policy.py` (Beobachtung, Aktion, Begrenzung, beide Modi) und
  `model/nova_env.py` (Gymnasium-Umgebung mit `reset`/`step`/Belohnung).
- Tests: `gymnasium.utils.env_checker.check_env`, Netz-Ausgabe 0 ergibt genau
  den klassischen Trab, Kinematik hin und zurück, Grenzen werden eingehalten.
- Ergebnis: Ein zufälliges Netz läuft durch, Korrektur 0 = Trab.
- **Stand 15.11.2026: erledigt** (lokal). `src/policy.py`, `model/nova_env.py`,
  `model/config.py`, dazu Fehler des Lagesensors in `randomization.py`.
  `check_env` ohne Warnung, Korrektur 0 ergibt exakt die Winkel des Trabs.
  Ohne Zufall trabt Korrektur 0 alle 10 s durch (87 % der befohlenen Strecke),
  mit Randomisierung und Stößen fallen 2 von 8 Episoden. Ein zufälliges Netz
  fällt meist um. Stufe 2: Den Trab nachmachen bringt 950 Punkte, nur Stehen
  617, Laufen lohnt sich also. Tempo am Laptop ca. 630 Schritte/s, am PC mit
  mehreren Kernen entsprechend mehr. Auffällig für Phase 3: Der Term
  „Drehrate halten“ gibt dem Trab nur 0,62 (der Rumpf schwingt beim Traben
  um die Hochachse), Stehen bekommt 1,0.

**Phase 3: Training Stufe 1**
- `model/train.py`: zuerst ohne Randomisierung (prüfen, ob es überhaupt lernt),
  dann Randomisierung schrittweise, dann Stöße.
- `model/export_onnx.py`, Vergleichstest PyTorch vs. ONNX.
- Abnahme (Vorschlag): Mit Randomisierung übersteht das Netz deutlich mehr
  Episoden als der Trab, fällt bei einem 15-N-Stoß nicht um, und die Drehung
  bei ±3° Kalibrierfehler sinkt mindestens auf die Hälfte.

- **Stand 18.11.2026: Code fertig** (lokal), Training am PC offen. `model/train.py`
  (Lehrplan 2 / 6 / 7 Mio. Schritte), `model/export_onnx.py` (Gegenprobe mit
  onnxruntime), `model/evaluate.py --policy` (Trab gegen Netz, prüft die
  Abnahme), `policy.OnnxNetwork` mit Prüfung der Einstellungen. Lerntest am
  Laptop (400.000 Schritte ohne Zufall): Das Netz lernt (Punkte 32 → 261).
  Zwei Änderungen gegenüber dem Plan: `log_std_init` −2,0 statt −1,0 (mit −1,0
  läuft der Trab beim Ausprobieren nur 12 % der Strecke, mit −2,0 noch 79 %)
  und ein zusätzlicher Belohnungs-Term „Kurs halten“ exp(−(Δψ / 10°)²), Gewicht
  +0,5, weil „Drehrate halten“ langsames Abdriften kaum bestraft (2 °/s kosten
  dort nur 2 %). Abnahme mit 20-N- statt 15-N-Stoß (siehe Phase 1).

**Phase 4: Stufe 2 und Pi-Software**
- `model/imitation.py`: Daten mit dem Lehrer sammeln, Behavior Cloning.
- `model/train.py --mode full`: PPO-Feinschliff mit ausgeblendeter
  Nachahmung.
- Servo-Messungen auswerten (`model/sysid.py`), Werte in die Simulation,
  Stufe 1 neu trainieren.
- `src/distance.py`, `src/controller.py` mit Fake-Modus, Software-in-the-Loop
  am PC (5.6). Am Pi: venv, Laufzeit der Schleife messen.
- Abnahme: Das Netz aus Stufe 2 läuft in der Simulation allein mindestens so
  gut wie der klassische Trab. Die ganze Kette läuft am PC gegen MuJoCo und
  am Pi ohne Servos.

**Phase 5: Echter Hund**
- Reihenfolge aus 5.5. Logs vergleichen, Modell anpassen, neu trainieren.
- Abnahme: Stufe 1 am echten Hund, gemessen gegen den Trab ohne Netz; Stufe 2
  zumindest aufgebockt und kurze Strecken am Boden.

**Phase 6: Kann-Ziele**
- Drehen auf Befehl, `src/obstacle.py` mit Regeln, optional Netz
  (`model/obstacle_env.py`).
- Abnahme: Nova läuft auf eine Wand zu und dreht ab.

**Phase 7: Auswertung und Arbeit**
- Diagramme aus `evaluate.py` und den echten Logs, Vergleich der drei
  Verfahren.

---

## 7. Ordnerstruktur und neue Dateien

```
RoboterHundDiplomarbeit/
├── docs/
│   └── neuronales-netz-plan.md   (neu)    diese Datei
├── src/                                   läuft am Pi; am PC mit NOVA_SIM=1
│   ├── servos.py          (ändern)        + Kalibrierung Kinematik-Winkel ↔ Servo-Winkel, Grenzen für beide
│   ├── kinematics.py      (wie bisher)
│   ├── gait.py            (ändern)        + Drehrate im Trab, Phase als Eigenschaft
│   ├── nova_model.py      (ändern)        + IMU, Ultraschall, Bodenreibung, Selbstkollision, Grenzen aus servos.py, Datenblattwerte
│   ├── sim.py             (ändern)        + Servo-Modell (Totzeit, Kennlinie, Totzone, PCA9685-Stufen), Sensoren auslesen
│   ├── imu.py             (neu)           MPU6050 lesen + Komplementärfilter; Filter läuft auch in der Sim; Fake-Modus
│   ├── policy.py          (neu)           Beobachtung bauen, ONNX rechnen, Ausgabe begrenzen; beide Stufen; gleich in Sim und am Pi
│   ├── controller.py      (neu)           50-Hz-Hauptschleife mit Sicherheitsabschaltung, Überblendung und Log
│   ├── distance.py        (neu, Phase 6)  Ultraschall; Fake-Modus
│   └── obstacle.py        (neu, Phase 6)  Regeln bzw. Netz: Abstand → Tempo, Drehrate
├── model/                                 nur PC: Training (Ordner steht schon im README)
│   ├── requirements.txt   (neu)           mujoco, gymnasium, stable-baselines3, torch, tensorboard, onnx, onnxruntime
│   ├── config.py          (neu)           alle Zahlen an einer Stelle: Belohnung, Randomisierung, PPO, beide Stufen
│   ├── nova_env.py        (neu)           Gymnasium-Umgebung auf Basis von NovaSim, mode="residual" oder "full"
│   ├── randomization.py   (neu)           Domain Randomization
│   ├── train.py           (neu)           PPO-Training, Checkpoints, TensorBoard
│   ├── imitation.py       (neu, Phase 4)  Daten vom Lehrer sammeln, Behavior Cloning für Stufe 2
│   ├── evaluate.py        (neu)           klassisch vs. Stufe 1 vs. Stufe 2, Kennzahlen, Diagramme
│   ├── export_onnx.py     (neu)           SB3 → ONNX + JSON, Vergleichstest
│   ├── sysid.py           (neu, Phase 4)  Servo-Messungen auswerten → Werte fürs Modell
│   ├── obstacle_env.py    (neu, optional) 2D-Umgebung für das Hindernis-Netz
│   └── policies/          (neu)           fertige Netze für den Pi: nova_residual_v1.onnx, nova_full_v1.onnx (+ .json)
├── tests/                 (neu)           pytest: Kinematik, Filter, Umgebung, ONNX-Gleichheit, Sicherheitsgrenzen
├── requirements-pc.txt    (neu)           matplotlib, mujoco, pytest (+ model/requirements.txt fürs Training)
└── requirements-pi.txt    (neu)           numpy, onnxruntime, adafruit-circuitpython-servokit, adafruit-circuitpython-mpu6050, gpiozero, lgpio
```

Die Skripte in `model/` importieren die Module aus `src/` über den Pfad
(`sys.path`), so wie die Module in `src/` sich heute schon gegenseitig
importieren. Ein eigenes Paket mit `pyproject.toml` wäre sauberer, ist aber für
den Anfang nicht nötig.

---

## 8. Risiken

| Risiko | Folge | Gegenmaßnahme |
|---|---|---|
| Betreuer erwartet etwas anderes | Ziel verfehlt | jetzt fragen (Text in 3.2) |
| Hund nicht bis Ende Jänner fertig | wenig Zeit für Phase 5 | Phasen 1 bis 4 laufen ohne Hardware; Software-in-the-Loop; Stufe 2 notfalls nur in der Simulation |
| Servos zu schwach für das echte Gewicht | Netz lernt Unmögliches, Servos überhitzen | früh wiegen und messen; Kennlinie im Modell; Strafe für Servo am Anschlag; leichter bauen, tiefer stehen |
| Stromversorgung zu schwach (Kabel, PCA9685-Klemme, LiPo) | Spannungseinbruch, Servos zucken, Pi stürzt ab | eigene Verteilung, dicke Kabel, Sicherung, Spannung mitloggen |
| Powerbank liefert zu wenig für den Pi 5 | Pi drosselt oder startet neu | Powerbank mit 5 V / 3 A Dauerleistung; mit `vcgencmd get_throttled` prüfen |
| MPU6050-Klon antwortet anders | Bibliothek verweigert den Dienst | `WHO_AM_I` prüfen, sonst `smbus2` direkt |
| Sim-Werte stimmen nicht | Netz funktioniert nur in der Sim | Messungen (5.1), breite Randomisierung, Logs vergleichen |
| Verzögerung am Pi größer als gedacht | Netz schaukelt sich auf | Totzeit messen, im Training abdecken, Netz-Anteil `k` |
| Stufe 2 überträgt sich nicht auf den echten Hund | Soll-Ziel nur in der Simulation | Stufe 1 als Muss-Ziel, Trab als Rückfallebene |
| Training lernt nichts | Zeitverlust | erst ohne Randomisierung, Belohnung einzeln prüfen, kleine Startausschläge, mehrere PCs parallel |
| Unterschiedliche Beobachtung in Sim und Pi | Netz macht am Pi Unsinn | gemeinsames `policy.py` und `imu.py`, Test mit aufgezeichneten Daten |

---

## 9. Offene Fragen

Geklärt: Abgabe ca. April 2027, Servos (Datenblatt oben), IMU MPU6050,
Training auf deinem PC (CPU).

1. **Betreuer:** Reicht "Netz verbessert den Trab" als Muss-Ziel, wenn "Netz
   läuft selbst" das Soll-Ziel ist? (Vorschlag für die Frage in 3.2)
2. **Dein PC:** Welche CPU (wie viele Kerne)? Davon hängt die Trainingsdauer ab.
3. **Hardware-Zeitplan:** Sind die Servos schon da (4 × 3er-Pack)? Wer druckt,
   und ist Ende Jänner für den fertigen Hund realistisch?
4. **Ultraschall (eilt nicht):** einer oder zwei Sensoren? Vorschlag: zwei
   RCWL-1601 an 3,3 V.

---

## 10. Zum Weiterlesen

- M. Rahme u. a.: *Dynamics and Domain Randomized Gait Modulation with Bezier
  Curves for Sim-to-Real Legged Locomotion* (2020), Code: GitHub
  `OpenQuadruped/spot_mini_mini`. Spot-Micro-Klon, klassischer Gang + gelernte
  Korrektur, auf echte Hardware übertragen. Das Vorbild für Stufe 1.
- A. Iscen u. a.: *Policies Modulating Trajectory Generators* (CoRL 2018).
  Die Idee hinter Stufe 1.
- X. B. Peng u. a.: *DeepMimic* (2018) und *Learning Agile Robotic Locomotion
  Skills by Imitating Animals* (RSS 2020). Nachahmen + RL, die Idee hinter
  Stufe 2, im zweiten Paper auf einem echten Vierbeiner.
- J. Tan u. a.: *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*
  (RSS 2018). Servo-Modell, Verzögerung und Randomisierung, gut erklärt.
- N. Rudin u. a.: *Learning to Walk in Minutes Using Massively Parallel Deep
  RL* (2021), Code: GitHub `leggedrobotics/legged_gym`. Vorlage für
  Belohnungsterme.
- Stable-Baselines3-Doku (PPO, "Exporting models" für ONNX), Gymnasium-Doku
  (eigene Umgebung), MuJoCo-Doku (Aktuatoren, Sensoren, Kontakt-Parameter),
  MPU6050-Registerbeschreibung (InvenSense) für `DLPF` und `WHO_AM_I`.
- MuJoCo Playground (GitHub `google-deepmind/mujoco_playground`):
  GPU-Training für Laufroboter, falls Stufe 2 auf der CPU zu langsam ist.
