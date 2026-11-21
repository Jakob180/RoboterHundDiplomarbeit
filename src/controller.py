"""
Hauptschleife von Nova (Plan 5.4 und 5.6): Lagesensor -> Filter -> Netz -> Gelenkwinkel -> Servos.

Läuft am Pi mit 50 Hz gegen die echten Servos und den MPU6050 und am PC mit
NOVA_SIM=1 gegen die MuJoCo-Simulation (Software-in-the-Loop). Beobachtung,
Netz-Ausgabe und Begrenzung kommen aus policy.py, der Lagefilter aus imu.py:
genau derselbe Code wie im Training.

    NOVA_SIM=1 python src/controller.py --policy model/policies/nova_full_v1.onnx --k 1 --speed 150
    python src/controller.py --hardware --policy model/policies/nova_residual_v1.onnx --k 0.25 --speed 100

Sicherheit (Plan 5.4):
- k (0 bis 1) blendet zwischen klassischem Trab (0) und Netz (1) über. Standard 0.
- Neigung über 30° -> Servos sofort locker, Schleife aus.
- Netz liefert NaN, Lagesensor antwortet nicht, Takt mehr als 3 Takte zu spät ->
  dieser Takt ohne Netz (klassischer Trab). Fehlt der Lagesensor 0,5 s lang: anhalten.
- Alle Winkel gehen über servos.py und werden dort auf die Gelenkgrenzen begrenzt.
- Echte Servos nur mit --hardware. Einschalten: Hund liegt, Liegepose wird gesendet,
  dann Servostrom an, dann langsam aufstehen.

Noch nicht auf der Hardware getestet.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from gait import LIE, STAND, Pose
from imu import ComplementaryFilter
from kinematics import body_ik
from policy import (
    CONTROL_HZ, FullMode, ResidualMode, clamp_angles, joint_array, joint_dict, mode_from_settings,
)
from servos import JOINT_ORDER, to_servo_angle

PERIOD = 1.0 / CONTROL_HZ   # s, 50 Hz wie im Training
TILT_LIMIT = 30.0           # Grad: darüber ist Nova am Umfallen
LATE_LIMIT = 3.0            # Takte: so viel zu spät läuft der Takt ohne Netz
IMU_LIMIT = 25              # Takte (0,5 s) ohne Lagesensor -> anhalten
MAX_STEP = 8.0              # Grad pro Takt beim Aufstehen und Hinlegen höchstens
SIM_MODE = os.environ.get("NOVA_SIM") == "1"
SETTLE = 10                 # Takte, die sich die Simulation am Anfang setzen darf (sonst 20° Fehlmessung)


class Cycle(NamedTuple):
    """Was in einem Takt passiert ist."""

    state: str                    # "laeuft", "umgefallen", "aus", ...
    k: float                      # tatsächlich verwendeter Netz-Anteil
    flags: tuple[str, ...]        # z. B. "imu_fehler", "netz_ungueltig", "zu_spaet", "anhalten"
    tilt: float                   # Grad Neigung laut Filter
    angles: np.ndarray | None     # gesendete Gelenkwinkel (Grad, JOINT_ORDER)


# ---------------------------------------------------------------------------
# Hardware: echt (Pi) oder Simulation (PC)
# ---------------------------------------------------------------------------

class PiHardware:
    """Echte Servos (PCA9685) und echter Lagesensor (MPU6050). Noch nicht auf der Hardware getestet."""

    def __init__(self, servos=None, imu=None) -> None:
        if servos is None:
            from servos import Servos

            servos = Servos()
        if imu is None:
            from mpu6050 import Mpu6050

            imu = Mpu6050()
            imu.calibrate()  # Nullpunkt des Gyroskops, Nova muss dabei still liegen
        self.servos, self.imu = servos, imu

    def read_imu(self):
        return self.imu.read()

    def send(self, angles, dt: float) -> None:
        """Kinematik-Winkel -> Servo-Winkel, begrenzt in servos.py."""
        for name in JOINT_ORDER:
            kinematic = getattr(angles[name[:2]], name[3:])
            self.servos.set_joint(name, to_servo_angle(name, kinematic))

    def relax(self) -> None:
        self.servos.relax()


class DryHardware:
    """Trockenlauf: keine Servos, Lagesensor meldet "steht gerade". Zum Messen der Rechenzeit (auch am Pi)."""

    def read_imu(self):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 9.81)

    def send(self, angles, dt: float) -> None:
        pass

    def relax(self) -> None:
        pass


class SimHardware:
    """Software-in-the-Loop: MuJoCo statt Servos und Lagesensor (nur PC). send() rechnet einen Takt Physik."""

    def __init__(self, start: Pose = STAND) -> None:
        from sim import NovaSim  # erst hier: MuJoCo gibt es am Pi nicht

        self.sim = NovaSim(self_collision=True)
        angles = body_ik(start.feet, start.body)
        self.sim.reset(angles, start.body.height)
        for _ in range(SETTLE):  # erst aufsetzen lassen: Der erste Messwert direkt nach dem Start ist verfälscht
            self.sim.step(angles)
        self.relaxed = False

    def read_imu(self):
        return self.sim.imu()

    def send(self, angles, dt: float) -> None:
        if not self.relaxed:
            self.sim.step(angles, dt)

    def relax(self) -> None:
        self.relaxed = True


# ---------------------------------------------------------------------------
# Regler
# ---------------------------------------------------------------------------

LOG_COLUMNS = (
    ["zeit", "dt", "zu_spaet", "zustand", "k", "tempo", "drehrate", "oben_x", "oben_y", "oben_z", "neigung",
     "gyro_x", "gyro_y", "gyro_z"]
    + [f"netz_{i}" for i in range(len(JOINT_ORDER))] + list(JOINT_ORDER) + ["hinweise"]
)


class Controller:
    """Ein Takt nach dem anderen: step() alle 20 ms aufrufen (macht run() am Pi)."""

    def __init__(self, hardware, network=None, mode=None, k: float = 0.0, log=None) -> None:
        self.hardware = hardware
        self.network = network
        if mode is None:
            settings = getattr(network, "settings", None)
            mode = mode_from_settings(settings) if settings else ResidualMode()
        self.mode = mode
        self.k = k
        self.filter = ComplementaryFilter()
        self.state = "bereit"
        self.time = 0.0
        self._gyro = (0.0, 0.0, 0.0)
        self._imu_failures = 0
        self._last = joint_array(body_ik(STAND.feet, STAND.body))  # zuletzt gesendete Winkel
        self._log_file = None
        if log is not None:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log, "w", newline="", encoding="utf-8")
            self._log = csv.writer(self._log_file)
            self._log.writerow(LOG_COLUMNS)

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = float(min(1.0, max(0.0, value)))

    def set_command(self, speed: float, yaw_rate: float) -> None:
        """Tempo in mm/s (+ = vorwärts), Drehrate in °/s (+ = links), wird weich angefahren."""
        self.mode.set_command(speed, yaw_rate)

    @property
    def tilt(self) -> float:
        """Neigung des Rumpfs laut Lagefilter in Grad."""
        return math.degrees(math.acos(max(-1.0, min(1.0, self.filter.up[2]))))

    # -----------------------------------------------------------------------

    def _sense(self, dt: float, flags: list[str]) -> bool:
        """Lagesensor lesen und filtern. False = Nova kippt (Servos sind dann schon locker)."""
        try:
            gyro, accel = self.hardware.read_imu()
            if not (np.all(np.isfinite(gyro)) and np.all(np.isfinite(accel))):
                raise ValueError("Lagesensor liefert ungültige Werte")
            self.filter.update(gyro, accel, dt)
            self._gyro = gyro
            self._imu_failures = 0
        except (OSError, ValueError):
            flags.append("imu_fehler")
            self._gyro = (0.0, 0.0, 0.0)
            self._imu_failures += 1
            if self._imu_failures == IMU_LIMIT:
                self.set_command(0.0, 0.0)
                flags.append("anhalten")
        if self.tilt > TILT_LIMIT:
            self.hardware.relax()
            self.state = "umgefallen"
            return False
        return True

    def _send(self, angles: np.ndarray, dt: float) -> None:
        self.hardware.send(joint_dict(angles), dt)
        self._last = angles

    def _write(self, dt, late, k, flags, output, angles) -> None:
        self.time += dt
        if self._log_file is None:
            return
        up, gyro = self.filter.up, self._gyro
        output = np.zeros(len(JOINT_ORDER)) if output is None else output
        angles = np.full(len(JOINT_ORDER), np.nan) if angles is None else angles
        self._log.writerow(
            [f"{self.time:.3f}", f"{dt:.4f}", f"{late:.2f}", self.state, f"{k:.3f}", f"{self.mode.speed:.1f}",
             f"{self.mode.yaw_rate:.1f}", *(f"{v:.4f}" for v in up), f"{self.tilt:.2f}", *(f"{v:.4f}" for v in gyro),
             *(f"{v:.4f}" for v in output), *(f"{v:.2f}" for v in angles), " ".join(flags)]
        )

    # -----------------------------------------------------------------------

    def step(self, dt: float = PERIOD, late: float = 0.0) -> Cycle:
        """Ein Takt. dt: echte Zeit seit dem letzten Takt (s), late: wie viele Takte zu spät."""
        if self.state in ("umgefallen", "aus"):
            return Cycle(self.state, 0.0, (), self.tilt, None)
        flags: list[str] = []
        if not self._sense(dt, flags):
            self._write(dt, late, 0.0, flags, None, None)
            return Cycle(self.state, 0.0, tuple(flags), self.tilt, None)

        k = self.k
        if late > LATE_LIMIT:
            flags.append("zu_spaet")
        if "imu_fehler" in flags or "zu_spaet" in flags:
            k = 0.0

        obs = self.mode.observe(self.filter.up, self._gyro)
        output = np.zeros(len(JOINT_ORDER))
        if self.network is not None:
            output = np.asarray(self.network(obs), dtype=float).reshape(-1)
            if output.shape != (len(JOINT_ORDER),) or not np.all(np.isfinite(output)):
                flags.append("netz_ungueltig")
                output, k = np.zeros(len(JOINT_ORDER)), 0.0

        # Stufe 1: Netz-Korrektur mal k (0 = Trab). Stufe 2: zwischen Trab und Netz überblenden.
        # Der Trab läuft im Takt des Trainings (PERIOD), nur der Filter bekommt die echte Zeit.
        if isinstance(self.mode, FullMode):
            self.mode.act(output, PERIOD)
            angles = clamp_angles((1.0 - k) * self.mode.reference + k * self.mode.angles)
        else:
            angles = joint_array(self.mode.act(k * output, PERIOD))

        self._send(angles, dt)
        self.state = "laeuft"
        self._write(dt, late, k, flags, output, angles)
        return Cycle(self.state, k, tuple(flags), self.tilt, angles)

    def _move(self, target: np.ndarray, seconds: float, state: str, dt: float = PERIOD) -> bool:
        """Von den zuletzt gesendeten Winkeln weich (S-Kurve) zu `target`. False = umgefallen."""
        start = self._last.copy()
        n = max(1, round(seconds / dt), math.ceil(np.max(np.abs(target - start)) * 1.5 / MAX_STEP))
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)  # langsam anfahren und abbremsen
            flags: list[str] = []
            if not self._sense(dt, flags):
                return False
            self.state = state
            angles = start + (target - start) * s
            self._send(angles, dt)
            self._write(dt, 0.0, 0.0, flags, None, angles)
        return True

    def hold(self, pose: Pose, seconds: float, dt: float = PERIOD) -> bool:
        """Eine Pose halten (z. B. zuerst die Liegepose senden). False = umgefallen."""
        angles = joint_array(body_ik(pose.feet, pose.body))
        for _ in range(max(1, round(seconds / dt))):
            flags: list[str] = []
            if not self._sense(dt, flags):
                return False
            self.state = "halten"
            self._send(angles, dt)
            self._write(dt, 0.0, 0.0, flags, None, angles)
        return True

    def stand_up(self, seconds: float = 2.0) -> bool:
        """Aus der Liegepose langsam in die Stehpose, danach beginnt der Trab dort."""
        ok = self._move(joint_array(body_ik(STAND.feet, STAND.body)), seconds, "aufstehen")
        self.mode.reset()
        self.state = "bereit" if ok else self.state
        return ok

    def lie_down(self, seconds: float = 2.0) -> None:
        """Anhalten, langsam in die Liegepose, dann Servos locker."""
        self.set_command(0.0, 0.0)
        if self.state != "umgefallen":
            self._move(joint_array(body_ik(LIE.feet, LIE.body)), seconds, "hinlegen")
        self.hardware.relax()
        if self.state != "umgefallen":
            self.state = "aus"

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------------------
# Echtzeit-Schleife und Kommandozeile
# ---------------------------------------------------------------------------

def run(
    controller: Controller, seconds: float, period: float = PERIOD, clock=time.perf_counter, sleep=time.sleep,
    timings: list | None = None,
) -> int:
    """`seconds` lang alle `period` s einen Takt. Gibt die Zahl der Takte zurück.

    timings: Liste, in die die Rechenzeit jedes Takts (s) geschrieben wird.
    """
    total = round(seconds / period)
    start = clock()
    last = start
    for n in range(total):
        now = clock()
        late = max(0.0, (now - (start + n * period)) / period)
        if late > LATE_LIMIT:
            start = now - n * period  # weit hinten: nicht alles aufholen, ab jetzt neu im Takt
        dt = now - last if n else period
        last = now
        controller.step(dt, late)
        if timings is not None:
            timings.append(clock() - now)
        if controller.state in ("umgefallen", "aus"):
            return n + 1
        wait = start + (n + 1) * period - clock()
        if wait > 0:
            sleep(wait)
    return total


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nova laufen lassen (am Pi oder mit NOVA_SIM=1 in der Simulation)")
    parser.add_argument("--policy", help="Netz (.onnx aus model/policies), ohne: klassischer Trab")
    parser.add_argument("--k", type=float, default=0.0, help="Netz-Anteil 0 (Trab) bis 1 (nur Netz), Standard 0")
    parser.add_argument("--speed", type=float, default=0.0, help="Tempo in mm/s, Standard 0 (auf der Stelle)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Drehrate in °/s (+ = links)")
    parser.add_argument("--seconds", type=float, default=10.0, help="so lange laufen")
    parser.add_argument("--log", help="Protokoll als CSV, z. B. logs/lauf1.csv")
    parser.add_argument("--hardware", action="store_true", help="echte Servos bewegen (nur am Pi)")
    parser.add_argument("--dry", action="store_true",
                        help="Trockenlauf ohne Servos und Sensor in Echtzeit: misst die Rechenzeit pro Takt")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry:
        return dry_run(args)
    if not SIM_MODE and not args.hardware:
        print("Achtung: Das bewegt die echten Servos. Erst mit --hardware, und nur wenn jemand am Not-Aus steht.\n"
              "Zum Testen am PC: NOVA_SIM=1 setzen (Simulation).")
        return 2
    network = None
    if args.policy:
        from policy import OnnxNetwork

        network = OnnxNetwork(args.policy)
    hardware = SimHardware(start=LIE) if SIM_MODE else PiHardware()
    controller = Controller(hardware, network, k=args.k, log=args.log)
    try:
        controller.hold(LIE, 0.5)
        if not SIM_MODE:
            input("Liegepose wird gesendet. Servostrom einschalten, dann Enter (Strg+C bricht ab) ... ")
        if not controller.stand_up():
            print("Beim Aufstehen umgekippt, Servos sind locker.")
            return 1
        controller.hold(STAND, 1.0)
        controller.set_command(args.speed, args.yaw)
        if SIM_MODE:
            cycles = 0
            for _ in range(round(args.seconds / PERIOD)):
                controller.step()
                cycles += 1
                if controller.state != "laeuft":
                    break
        else:
            cycles = run(controller, args.seconds)
        state = controller.state
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        state = "abgebrochen"
        cycles = 0
    finally:
        controller.lie_down()
        controller.close()
    print(f"{cycles} Takte, Zustand am Ende: {state}")
    if SIM_MODE:
        torso = hardware.sim.torso()
        print(f"Strecke {torso.x:.0f} mm, seitlich {torso.y:+.0f} mm, gedreht {torso.yaw:+.1f}°")
    if args.log:
        print(f"Protokoll: {args.log}")
    return 0 if state in ("laeuft", "aus") else 1


def dry_run(args) -> int:
    """Ganze Schleife in Echtzeit ohne Hardware: Wie lange rechnet ein Takt? (Plan Phase 4, auch am Pi)"""
    network = None
    if args.policy:
        from policy import OnnxNetwork

        network = OnnxNetwork(args.policy)
    controller = Controller(DryHardware(), network, k=args.k, log=args.log)
    controller.set_command(args.speed, args.yaw)
    timings: list[float] = []
    cycles = run(controller, args.seconds, timings=timings)
    controller.close()
    ms = np.array(timings) * 1000
    print(f"{cycles} Takte. Rechenzeit pro Takt: Mittel {ms.mean():.2f} ms, 99 % unter {np.percentile(ms, 99):.2f} ms, "
          f"höchstens {ms.max():.2f} ms (Takt: {PERIOD * 1000:.0f} ms)")
    late = int(np.sum(ms > PERIOD * 1000))
    print(f"Takte zu spät: {late}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
