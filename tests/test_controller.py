"""Hauptschleife für den Pi, getestet gegen die Simulation (Software-in-the-Loop) und mit Attrappen."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import controller
import gait
import policy
from imu import G
from kinematics import body_ik

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "model" / "policies"
LEVEL = ((0.0, 0.0, 0.0), (0.0, 0.0, G))   # Drehrate, Beschleunigung: steht gerade und ruhig


class FakeHardware:
    """Merkt sich, was gesendet wird. Lagesensor: fest vorgegeben oder kaputt."""

    def __init__(self, imu=LEVEL, broken=False) -> None:
        self.imu, self.broken = imu, broken
        self.sent: list[np.ndarray] = []
        self.relaxed = False

    def read_imu(self):
        if self.broken:
            raise OSError("I2C antwortet nicht")
        return self.imu

    def send(self, angles, dt) -> None:
        self.sent.append(policy.joint_array(angles))

    def relax(self) -> None:
        self.relaxed = True


def trot_angles(steps: int, speed: float = 150.0) -> list[np.ndarray]:
    m = policy.ResidualMode()
    m.set_command(speed, 0.0)
    return [policy.joint_array(m.act(np.zeros(12), controller.PERIOD)) for _ in range(steps)]


def tilted(degrees: float):
    a = math.radians(degrees)
    return (0.0, 0.0, 0.0), (G * math.sin(a), 0.0, G * math.cos(a))


# ---------------------------------------------------------------------------
# Ablauf eines Takts
# ---------------------------------------------------------------------------

def test_without_network_the_controller_trots():
    hw = FakeHardware()
    c = controller.Controller(hw)
    c.set_command(150.0, 0.0)
    for _ in range(60):
        c.step()
    assert np.allclose(hw.sent, trot_angles(60))


def test_k_zero_means_the_network_has_no_say():
    for mode in (policy.ResidualMode(), policy.FullMode()):
        hw = FakeHardware()
        c = controller.Controller(hw, network=lambda obs: np.ones(12), mode=mode, k=0.0)
        c.set_command(150.0, 0.0)
        for _ in range(60):
            c.step()
        assert np.allclose(hw.sent, trot_angles(60)), mode.name


def test_k_one_lets_the_network_steer():
    hw = FakeHardware()
    c = controller.Controller(hw, network=lambda obs: np.ones(12), mode=policy.ResidualMode(), k=1.0)
    c.set_command(150.0, 0.0)
    for _ in range(60):
        c.step()
    assert not np.allclose(hw.sent[-1], trot_angles(60)[-1])


def test_stage_2_blends_from_the_trot_into_the_network():
    def run(k):
        hw = FakeHardware()
        c = controller.Controller(hw, network=lambda obs: np.full(12, 0.5), mode=policy.FullMode(), k=k)
        c.set_command(150.0, 0.0)
        for _ in range(60):
            c.step()
        return hw.sent[-1]
    trot, half, net = run(0.0), run(0.5), run(1.0)
    assert half == pytest.approx((trot + net) / 2)


def test_k_is_limited_to_zero_to_one():
    c = controller.Controller(FakeHardware(), k=3.0)
    assert c.k == 1.0
    c.k = -1.0
    assert c.k == 0.0


# ---------------------------------------------------------------------------
# Sicherheit
# ---------------------------------------------------------------------------

def test_invalid_network_output_falls_back_to_the_trot():
    hw = FakeHardware()
    c = controller.Controller(hw, network=lambda obs: np.full(12, np.nan), mode=policy.FullMode(), k=1.0)
    c.set_command(150.0, 0.0)
    cycle = c.step()
    assert "netz_ungueltig" in cycle.flags and cycle.k == 0.0
    assert np.allclose(hw.sent[0], trot_angles(1)[0])


def test_tipping_over_relaxes_the_servos_and_stops():
    hw = FakeHardware(imu=tilted(35.0))
    c = controller.Controller(hw)
    cycle = c.step()
    assert cycle.state == "umgefallen" and hw.relaxed and hw.sent == []
    c.step()
    assert hw.sent == []  # bleibt aus


def test_slight_tilt_is_fine():
    hw = FakeHardware(imu=tilted(20.0))
    assert controller.Controller(hw).step().state == "laeuft" and not hw.relaxed


def test_missing_imu_trots_on_without_network_then_stands_still():
    hw = FakeHardware(broken=True)
    c = controller.Controller(hw, network=lambda obs: np.ones(12), mode=policy.ResidualMode(), k=1.0)
    c.set_command(150.0, 0.0)
    first = c.step()
    assert "imu_fehler" in first.flags and first.k == 0.0
    for _ in range(controller.IMU_LIMIT):
        cycle = c.step()
    assert "anhalten" in cycle.flags or c.mode.target_speed == 0.0
    assert c.mode.target_speed == 0.0 and c.mode.target_yaw_rate == 0.0


def test_late_loop_runs_this_cycle_without_network():
    c = controller.Controller(FakeHardware(), network=lambda obs: np.ones(12), mode=policy.ResidualMode(), k=1.0)
    assert c.step(late=1.0).k == 1.0
    cycle = c.step(late=controller.LATE_LIMIT + 1)
    assert "zu_spaet" in cycle.flags and cycle.k == 0.0


# ---------------------------------------------------------------------------
# Aufstehen, Hinlegen, Protokoll
# ---------------------------------------------------------------------------

def test_standing_up_starts_exactly_in_the_lying_pose_and_ends_standing():
    hw = FakeHardware()
    c = controller.Controller(hw)
    c.hold(gait.LIE, 0.1)
    c.stand_up(1.0)
    lie = policy.joint_array(body_ik(gait.LIE.feet, gait.LIE.body))
    assert np.allclose(hw.sent[0], lie)
    assert np.allclose(hw.sent[-1], policy.STAND_ANGLES)
    steps = np.abs(np.diff(np.array(hw.sent), axis=0)).max()
    assert steps < 8.0  # nie mehr als 8° pro Takt


def test_lying_down_ends_relaxed():
    hw = FakeHardware()
    c = controller.Controller(hw)
    c.set_command(150.0, 0.0)
    for _ in range(30):
        c.step()
    c.lie_down(1.0)
    lie = policy.joint_array(body_ik(gait.LIE.feet, gait.LIE.body))
    assert np.allclose(hw.sent[-1], lie) and hw.relaxed and c.state == "aus"


def test_every_cycle_is_logged(tmp_path):
    log = tmp_path / "lauf.csv"
    c = controller.Controller(FakeHardware(), log=log)
    c.set_command(100.0, 0.0)
    for _ in range(10):
        c.step()
    c.close()
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == 10
    assert float(rows[-1]["tempo"]) > 0 and rows[-1]["zustand"] == "laeuft"
    assert {"zeit", "k", "oben_z", "netz_0", "FL_upper", "hinweise"} <= set(rows[0])


# ---------------------------------------------------------------------------
# Echtzeit-Schleife
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, work: float = 0.0) -> None:
        self.now, self.work = 0.0, work

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_realtime_loop_runs_fifty_cycles_per_second():
    clock = FakeClock()
    c = controller.Controller(FakeHardware())
    cycles = controller.run(c, 2.0, clock=clock, sleep=clock.sleep)
    assert cycles == 100


def test_realtime_loop_notices_when_it_is_late():
    clock = FakeClock()
    hw = FakeHardware()
    c = controller.Controller(hw)
    original = c.step
    lates = []

    def slow_step(dt=controller.PERIOD, late=0.0):
        lates.append(late)
        if len(lates) == 5:
            clock.now += 0.11  # ein Takt hängt 110 ms
        return original(dt, late)
    c.step = slow_step
    controller.run(c, 0.4, clock=clock, sleep=clock.sleep)
    assert max(lates) >= 4.0


# ---------------------------------------------------------------------------
# Software-in-the-Loop: dieselbe Schleife gegen MuJoCo
# ---------------------------------------------------------------------------

def test_trot_walks_in_the_simulation():
    hw = controller.SimHardware()
    c = controller.Controller(hw)
    c.set_command(150.0, 0.0)
    for _ in range(250):
        cycle = c.step()
    torso = hw.sim.torso()
    assert cycle.state == "laeuft"
    assert torso.x > 0.75 * 150.0 * 5.0 and abs(torso.roll) < 10 and abs(torso.pitch) < 10


def test_standing_up_from_the_floor_in_the_simulation():
    hw = controller.SimHardware(start=gait.LIE)
    c = controller.Controller(hw)
    c.hold(gait.LIE, 0.5)
    c.stand_up(2.0)
    c.hold(gait.STAND, 1.0)
    assert hw.sim.torso().z == pytest.approx(gait.STAND.body.height, abs=20.0)


def test_strong_push_in_the_simulation_relaxes_the_servos():
    hw = controller.SimHardware()
    c = controller.Controller(hw)
    c.set_command(150.0, 0.0)
    for i in range(200):
        if i == 50:
            hw.sim.push((0.0, 60.0, 0.0), 0.3)
        cycle = c.step()
        if cycle.state != "laeuft":
            break
    assert cycle.state == "umgefallen" and hw.relaxed


@pytest.mark.parametrize("name", ["nova_residual_v1", "nova_full_v1"])
def test_trained_networks_walk_through_the_pi_loop(name):
    pytest.importorskip("onnxruntime")
    net = policy.OnnxNetwork(POLICIES / f"{name}.onnx")
    hw = controller.SimHardware()
    c = controller.Controller(hw, network=net, k=1.0)
    c.hold(gait.STAND, 0.5)
    c.set_command(150.0, 0.0)
    for _ in range(500):
        cycle = c.step()
    assert cycle.state == "laeuft" and not cycle.flags
    assert hw.sim.torso().x > 0.85 * 150.0 * 10.0


def test_pi_loop_walks_like_the_training_environment():
    """Gleicher Code in Sim und Pi: Schleife und Trainingsumgebung kommen gleich weit."""
    pytest.importorskip("onnxruntime")
    import randomization as rnd
    from nova_env import NovaEnv
    net = policy.OnnxNetwork(POLICIES / "nova_residual_v1.onnx")
    env = NovaEnv(mode="residual", randomize=False, pushes=False)
    obs, _ = env.reset(seed=0, options={"randomization": rnd.Randomization(), "command": (150.0, 0.0)})
    for _ in range(250):
        obs, *_ = env.step(net(obs))
    hw = controller.SimHardware()
    c = controller.Controller(hw, network=net, k=1.0)
    c.hold(gait.STAND, 0.5)
    c.set_command(150.0, 0.0)
    for _ in range(250):
        c.step()
    a, b = env.sim.torso(), hw.sim.torso()
    assert b.x == pytest.approx(a.x, rel=0.03)
    assert b.yaw == pytest.approx(a.yaw, abs=2.0)


# ---------------------------------------------------------------------------
# Echte Hardware (mit Attrappen) und Kommandozeile
# ---------------------------------------------------------------------------

def test_pi_hardware_turns_joint_angles_into_servo_angles():
    import servos
    from test_mpu6050 import FakeBus
    import mpu6050
    hw = controller.PiHardware(servos=servos.Servos(), imu=mpu6050.Mpu6050(bus=FakeBus()))
    angles = policy.joint_dict(policy.STAND_ANGLES)
    hw.send(angles, controller.PERIOD)
    for name, kin in zip(servos.JOINT_ORDER, policy.STAND_ANGLES):
        joint = servos.JOINTS[name]
        expected = servos.Servos.clamp(name, servos.to_servo_angle(name, kin))
        got = hw.servos._servo[joint.channel].angle
        assert got == pytest.approx(180.0 - expected if joint.inverted else expected)
    gyro, accel = hw.read_imu()
    assert accel[2] == pytest.approx(G, abs=1e-3)


def test_real_servos_only_with_hardware_flag(monkeypatch, capsys):
    monkeypatch.setattr(controller, "SIM_MODE", False)
    assert controller.main(["--seconds", "1"]) == 2
    assert "--hardware" in capsys.readouterr().out


def test_command_line_runs_in_the_simulation(tmp_path, capsys):
    log = tmp_path / "lauf.csv"
    assert controller.main(["--seconds", "2", "--speed", "100", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "Strecke" in out and log.exists()


def test_simulation_has_settled_before_the_first_imu_reading():
    for pose in (gait.LIE, gait.STAND):
        c = controller.Controller(controller.SimHardware(start=pose))
        c.hold(pose, controller.PERIOD)
        assert c.tilt < 3.0, pose


def test_pi_hardware_measures_the_gyro_zero_point_at_start(monkeypatch):
    import servos
    import mpu6050
    calls = []

    class FakeSensor:
        def calibrate(self):
            calls.append("calibrate")
    monkeypatch.setattr(mpu6050, "Mpu6050", FakeSensor)
    controller.PiHardware(servos=servos.Servos())
    assert calls == ["calibrate"]


def test_dry_run_measures_the_computing_time(capsys):
    pytest.importorskip("onnxruntime")
    code = controller.main(["--dry", "--seconds", "0.5", "--policy", str(POLICIES / "nova_full_v1.onnx"), "--k", "1"])
    out = capsys.readouterr().out
    assert code == 0 and "Rechenzeit" in out and "zu spät" in out


def test_realtime_loop_can_report_computing_times():
    clock = FakeClock()
    timings: list[float] = []
    controller.run(controller.Controller(FakeHardware()), 0.2, clock=clock, sleep=clock.sleep, timings=timings)
    assert len(timings) == 10
