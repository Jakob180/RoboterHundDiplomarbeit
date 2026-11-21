"""Lagesensor MPU6050 (GY-521) über I2C, getestet mit einem nachgebauten Bus."""

import math

import pytest

import mpu6050

G = 9.81


class FakeBus:
    """Tut so, als wäre ein MPU6050 am I2C-Bus. Rohwerte setzt der Test mit set_raw()."""

    def __init__(self, who_am_i: int = 0x68) -> None:
        self.written: dict[int, int] = {}
        self.who_am_i = who_am_i
        self.raw = [0, 0, 8192, 0, 0, 0, 0]  # ax, ay, az, Temperatur, gx, gy, gz
        self.reads = 0

    def set_raw(self, ax=0, ay=0, az=8192, gx=0, gy=0, gz=0) -> None:
        self.raw = [ax, ay, az, 0, gx, gy, gz]

    def write_byte_data(self, address, register, value) -> None:
        assert address == 0x68
        self.written[register] = value

    def read_byte_data(self, address, register) -> int:
        assert address == 0x68 and register == 0x75
        return self.who_am_i

    def read_i2c_block_data(self, address, register, length) -> list[int]:
        assert address == 0x68 and register == 0x3B and length == 14
        self.reads += 1
        data = []
        for value in self.raw:
            value &= 0xFFFF  # Zweierkomplement wie am Sensor
            data += [value >> 8, value & 0xFF]
        return data


def sensor(**options) -> tuple[mpu6050.Mpu6050, FakeBus]:
    bus = FakeBus(**options)
    return mpu6050.Mpu6050(bus=bus), bus


def test_sensor_is_woken_up_and_set_to_its_ranges():
    _, bus = sensor()
    assert bus.written[0x6B] == 0x01          # aufwecken, Takt vom Gyroskop
    assert bus.written[0x1B] == 0x08          # Gyroskop ±500 °/s
    assert bus.written[0x1C] == 0x08          # Beschleunigung ±4 g
    assert bus.written[0x1A] == 0x03          # Tiefpass ca. 44 Hz


def test_level_sensor_measures_one_g_upwards():
    imu, _ = sensor()
    gyro, accel = imu.read()
    assert accel == pytest.approx((0.0, 0.0, G), abs=1e-3)
    assert gyro == pytest.approx((0.0, 0.0, 0.0))


def test_raw_values_become_rad_per_second_and_metres_per_second_squared():
    imu, bus = sensor()
    bus.set_raw(ax=-4096, az=8192, gx=655, gz=-6550)  # -0,5 g; 10 °/s; -100 °/s
    gyro, accel = imu.read()
    assert accel == pytest.approx((-0.5 * G, 0.0, G), abs=1e-3)
    assert gyro[0] == pytest.approx(math.radians(10.0), rel=1e-3)
    assert gyro[2] == pytest.approx(math.radians(-100.0), rel=1e-3)


def test_unknown_chip_is_refused():
    with pytest.raises(OSError, match="0x12"):
        sensor(who_am_i=0x12)


@pytest.mark.parametrize("clone", [0x68, 0x70, 0x72, 0x98])
def test_common_clones_are_accepted(clone):
    sensor(who_am_i=clone)


def test_mounting_turns_the_axes_into_body_axes():
    bus = FakeBus()
    turned = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # Sensor um 90° gedreht eingebaut
    imu = mpu6050.Mpu6050(bus=bus, mount=turned)
    bus.set_raw(ax=8192, az=0)                   # 1 g entlang Sensor-x ...
    _, accel = imu.read()
    assert accel == pytest.approx((0.0, G, 0.0), abs=1e-3)  # ... ist Rumpf-y


def test_gyro_zero_point_is_measured_while_standing_still():
    imu, bus = sensor()
    bus.set_raw(gx=131, gy=-66, gz=20)           # Nullpunktfehler, Hund steht still
    imu.calibrate(samples=50, sleep=lambda s: None)
    gyro, _ = imu.read()
    assert gyro == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert bus.reads == 51


def test_calibration_refuses_a_moving_dog():
    imu, bus = sensor()
    values = iter(range(0, 50 * 400, 400))

    def moving(seconds):
        bus.set_raw(gx=next(values))             # dreht sich während der Messung
    with pytest.raises(RuntimeError, match="still"):
        imu.calibrate(samples=50, sleep=moving)
