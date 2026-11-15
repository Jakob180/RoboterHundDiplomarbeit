"""
Lage des Rumpfs aus dem Lagesensor (GY-521 mit MPU6050).

Der Sensor liefert zwei Dinge, die beide allein nicht reichen:
- Gyroskop (Drehrate): schnell und genau, läuft aber mit der Zeit davon (Drift).
- Beschleunigungssensor: zeigt in Ruhe, wo "oben" ist, zappelt aber bei jedem Schritt.

Der Komplementärfilter nimmt für schnelle Bewegungen das Gyroskop und zieht das
Ergebnis langsam zum Beschleunigungssensor hin. So bleibt die Lage ruhig und
läuft trotzdem nicht davon.

Dieselbe Klasse rechnet in der Simulation (mit den Sensorwerten aus MuJoCo) und
später am Pi (mit den echten Werten). Hier steht nur die Rechnung, das Auslesen
des Sensors über I2C kommt dazu, sobald er angeschlossen ist.

Einheiten wie am Sensor: Drehrate in rad/s, Beschleunigung in m/s².
Achsen wie überall: x vorne, y links, z oben.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]

G = 9.81  # m/s², Erdbeschleunigung


class ComplementaryFilter:
    """Schätzt, wo für den Rumpf "oben" ist. `update()` bei jedem Messwert aufrufen."""

    def __init__(self, time_constant: float = 0.5, accel_window: float = 0.15) -> None:
        # So schnell (in s) setzt sich der Beschleunigungssensor gegen das Gyroskop durch.
        # PLATZHALTER: klein = folgt schneller, zappelt aber mehr.
        self.time_constant = time_constant
        # Beschleunigung nur verwenden, wenn sie höchstens so viel (Anteil) von 1 g abweicht.
        # Mit 0,15 fallen die Stöße beim Aufsetzen der Füße im Trab heraus (in der Simulation
        # bis ca. 1,15 g). Mit 0,3 war der Filter im Trab bis 3,1° daneben, mit 0,15 bis 1,9°.
        self.accel_window = accel_window
        # Richtung "oben" aus Sicht des Rumpfs (Länge 1). Steht der Hund gerade: (0, 0, 1).
        self.up: Vec3 = (0.0, 0.0, 1.0)
        self._started = False

    def update(self, gyro: Vec3, accel: Vec3, dt: float) -> Vec3:
        """Neuen Messwert einrechnen: gyro in rad/s, accel in m/s², dt in s. Gibt `up` zurück."""
        # Dem Beschleunigungssensor nur trauen, wenn er ungefähr 1 g misst. Bei Stößen
        # (Fuß setzt auf) oder im Fall zeigt er nicht nach oben.
        size = math.sqrt(sum(a * a for a in accel))
        near_1g = abs(size - G) < self.accel_window * G
        measured = tuple(a / size for a in accel) if near_1g else None

        if measured and not self._started:
            self._started = True  # erster Messwert: Lage direkt übernehmen
            self.up = measured
            return self.up

        # Gyroskop: Dreht sich der Rumpf, wandert "oben" aus seiner Sicht in die Gegenrichtung
        ux, uy, uz = self.up
        wx, wy, wz = gyro
        new = (
            ux + (uy * wz - uz * wy) * dt,
            uy + (uz * wx - ux * wz) * dt,
            uz + (ux * wy - uy * wx) * dt,
        )
        # Beschleunigungssensor: ein kleines Stück in seine Richtung ziehen
        if measured:
            k = dt / (self.time_constant + dt)
            new = tuple(n + (m - n) * k for n, m in zip(new, measured))

        length = math.sqrt(sum(n * n for n in new))
        self.up = tuple(n / length for n in new)
        return self.up

    @property
    def roll(self) -> float:
        """Seitliche Neigung in Grad (+ = linke Seite oben)."""
        return math.degrees(math.atan2(self.up[1], self.up[2]))

    @property
    def pitch(self) -> float:
        """Neigung nach vorne in Grad (+ = Nase runter)."""
        return math.degrees(math.asin(max(-1.0, min(1.0, -self.up[0]))))
