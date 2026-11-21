"""
Lagesensor GY-521 (MPU6050) über I2C auslesen, nur am Pi.

Liefert Drehrate (rad/s) und Beschleunigung (m/s²) in Rumpf-Achsen (x vorne,
y links, z oben), genau wie `NovaSim.imu()` in der Simulation. Die Rechnung mit
dem Komplementärfilter steht in imu.py und ist für Sim und Pi dieselbe.

    imu = Mpu6050()        # öffnet I2C-Bus 1, Adresse 0x68
    imu.calibrate()        # Hund muss dabei still stehen (Nullpunkt des Gyroskops)
    gyro, accel = imu.read()

Braucht am Pi `smbus2` im venv. Noch nicht am echten Sensor getestet.
"""

from __future__ import annotations

import math
import time

import numpy as np

G = 9.81
ADDRESS = 0x68               # AD0 auf GND (Standard beim GY-521)

# Register (Datenblatt "MPU-6000/MPU-6050 Register Map")
CONFIG = 0x1A                # digitaler Tiefpass
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B          # ab hier 14 Byte: Beschleunigung xyz, Temperatur, Drehrate xyz
PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75

# Billige GY-521 haben oft einen Nachbau-Chip, der sich anders meldet, aber gleich ausgelesen wird
KNOWN_CHIPS = {0x68: "MPU6050", 0x70: "MPU6500", 0x72: "Nachbau", 0x98: "Nachbau"}

DLPF = 0x03                  # Tiefpass ca. 44 Hz, ca. 5 ms Verzögerung (Training deckt 0 bis 10 ms ab)
GYRO_RANGE = 0x08            # ±500 °/s
GYRO_SCALE = 65.5            # LSB pro °/s bei ±500 °/s
ACCEL_RANGE = 0x08           # ±4 g (Aufsetzen der Füße bis ca. 2 g)
ACCEL_SCALE = 8192.0         # LSB pro g bei ±4 g

# Einbaulage: Rumpf-Achsen = MOUNT · Sensor-Achsen. PLATZHALTER: Sensor gerade eingebaut,
# x nach vorne, z nach oben. Am echten Hund prüfen (Hund nach vorne kippen -> Drehrate um y).
MOUNT = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

STILL = math.radians(1.0)    # rad/s: so viel darf das Gyroskop beim Nullpunkt-Abgleich schwanken


def _signed(high: int, low: int) -> int:
    value = (high << 8) | low
    return value - 0x10000 if value & 0x8000 else value


class Mpu6050:
    def __init__(self, bus=None, address: int = ADDRESS, mount=MOUNT) -> None:
        if bus is None:
            from smbus2 import SMBus  # erst hier, damit die Datei auch am PC ohne smbus2 geht

            bus = SMBus(1)
        self.bus, self.address = bus, address
        self.mount = np.asarray(mount, dtype=float)
        self.gyro_bias = np.zeros(3)  # rad/s, aus calibrate()
        chip = bus.read_byte_data(address, WHO_AM_I)
        if chip not in KNOWN_CHIPS:
            raise OSError(f"Unbekannter Lagesensor an 0x{address:02x}: WHO_AM_I = 0x{chip:02x}")
        self.chip = KNOWN_CHIPS[chip]
        bus.write_byte_data(address, PWR_MGMT_1, 0x01)  # aufwecken, Takt vom Gyroskop (genauer)
        bus.write_byte_data(address, CONFIG, DLPF)
        bus.write_byte_data(address, GYRO_CONFIG, GYRO_RANGE)
        bus.write_byte_data(address, ACCEL_CONFIG, ACCEL_RANGE)

    def _raw(self) -> tuple[np.ndarray, np.ndarray]:
        """(Drehrate in rad/s, Beschleunigung in m/s²) in Sensor-Achsen, ohne Nullpunkt-Abgleich."""
        d = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
        v = [_signed(d[i], d[i + 1]) for i in range(0, 14, 2)]
        accel = np.array(v[0:3], dtype=float) / ACCEL_SCALE * G
        gyro = np.radians(np.array(v[4:7], dtype=float) / GYRO_SCALE)
        return gyro, accel

    def read(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(Drehrate in rad/s, Beschleunigung in m/s²) in Rumpf-Achsen, Nullpunkt abgezogen."""
        gyro, accel = self._raw()
        gyro = self.mount @ gyro - self.gyro_bias
        accel = self.mount @ accel
        return tuple(float(g) for g in gyro), tuple(float(a) for a in accel)

    def calibrate(self, samples: int = 200, sleep=time.sleep, period: float = 0.005) -> None:
        """Nullpunkt des Gyroskops messen (ca. 1 s). Der Hund muss dabei still stehen."""
        rates = []
        for _ in range(samples):
            gyro, _ = self._raw()
            rates.append(self.mount @ gyro)
            sleep(period)
        rates = np.array(rates)
        if np.max(np.std(rates, axis=0)) > STILL:
            raise RuntimeError("Nova muss beim Nullpunkt-Abgleich still stehen, das Gyroskop hat sich bewegt")
        self.gyro_bias = rates.mean(axis=0)
