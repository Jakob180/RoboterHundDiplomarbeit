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
