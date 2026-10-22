# 🐕 Nova – Vierbeiniger Roboterhund

**Nova** ist ein vierbeiniger Roboterhund, den wir im Rahmen unserer Diplomarbeit an der **HTL Hallein** entwickeln. Ziel ist ein autonomer Laufroboter, der mithilfe eines neuronalen Netzes gesteuert wird und mit Akku unabhängig vom Stromnetz läuft.

## Überblick

| | |
|---|---|
| **Steuerung** | Raspberry Pi 5 |
| **Antrieb** | 12 Servomotoren (3 pro Bein) |
| **Mechanik** | Spot-Micro-Klon (3D-gedruckt, Vorlage von Thingiverse) |
| **Software** | Neuronales Netz für Bewegung und Steuerung |
| **Stromversorgung** | Akkubetrieb |

## Projektstatus

- [x] Planung, Teileliste und Budget
- [ ] Einrichtung Raspberry Pi 5 (Betriebssystem, SSH)
- [ ] Druck und Zusammenbau der Mechanik
- [ ] Ansteuerung der Servos
- [ ] Bewegungsabläufe (Stehen, Gehen)
- [ ] Training und Integration des neuronalen Netzes
- [ ] Akkubetrieb und Tests

## Projektstruktur

```
nova/
├── docs/        # Dokumentation, Diplomarbeit, Teileliste
├── hardware/    # CAD- und STL-Dateien, Schaltpläne
├── src/         # Quellcode (Steuerung, Servos, Sensoren)
├── model/       # Neuronales Netz: Training und Modelle
└── README.md
```

## Installation

> Folgt, sobald die erste lauffähige Version steht.

## Team

- Name – Aufgabenbereich
- Name – Aufgabenbereich
- Name – Aufgabenbereich

Betreuung: *Name der Lehrperson*

## Credits

Das mechanische Design basiert auf dem **Spot Micro**-Projekt bzw. einem Klon davon auf Thingiverse. Die Rechte am Design liegen bei den jeweiligen Urheber:innen.

---

*Diplomarbeit an der HTL Hallein, 2026/27*
