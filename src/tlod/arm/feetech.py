"""Real hardware backend: Feetech STS3215 bus servos over a serial adapter.

Talks to the servos directly through the Feetech SDK rather than going
through LeRobot. Two reasons:

1. Latency. This robot's whole problem is a ~300 ms sense-to-motion budget.
   Direct GroupSyncRead/GroupSyncWrite is one bus transaction per loop with
   nothing between us and the wire, and lets us set Goal_Acceleration and
   Goal_Speed per joint, which is how you actually make a fast strike.
2. Weight. `lerobot` depends on torch. Nothing in the control loop needs a
   deep learning framework, and on a Jetson the install is not free.

LeRobot interoperability is kept where it is genuinely useful: this class
reads calibration written by `lerobot-calibrate`, so the standard tooling
for homing offsets and joint ranges still works. Use whichever you prefer.

Register addresses are from the STS3215 control table; see
docs/hardware.md for the full map and the source it was checked against.

  !! UNVERIFIED AGAINST HARDWARE !!
  Written from the datasheet while the arm was still in shipping. Every
  path here is exercised by tests against a fake serial bus, but sign
  conventions and the calibration mapping must be confirmed with
  `tlod arm first-light` before this drives anything with torque on.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from tlod.arm.backend import ArmBackend
from tlod.arm.model import GRIPPER_LIMITS, JOINT_LIMITS
from tlod.types import JOINT_NAMES, NUM_JOINTS, JointState

log = logging.getLogger(__name__)

# STS3215 control table.
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

COUNTS_PER_REV = 4096
RAD_PER_COUNT = 2.0 * np.pi / COUNTS_PER_REV
CENTER_COUNT = 2048

# Motor ids 1..6 in JOINT_NAMES order, as set by `lerobot-setup-motors`.
MOTOR_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


class Calibration:
    """Per-joint mapping between encoder counts and radians.

    `center` is the count reading when the joint is at its zero angle;
    `sign` is +1 or -1 depending on whether the servo's positive direction
    matches the URDF joint axis.
    """

    def __init__(self, center: np.ndarray | None = None, sign: np.ndarray | None = None) -> None:
        self.center = np.full(NUM_JOINTS, float(CENTER_COUNT)) if center is None else np.asarray(center, float)
        self.sign = np.ones(NUM_JOINTS) if sign is None else np.asarray(sign, float)

    def to_rad(self, counts: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(counts, float) - self.center) * RAD_PER_COUNT

    def to_counts(self, rad: np.ndarray) -> np.ndarray:
        c = self.center + self.sign * np.asarray(rad, float) / RAD_PER_COUNT
        return np.clip(np.round(c), 0, COUNTS_PER_REV - 1).astype(int)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {n: {"center": float(c), "sign": float(s)}
                 for n, c, s in zip(JOINT_NAMES, self.center, self.sign)},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Calibration":
        data = json.loads(Path(path).read_text())
        if all(k in data for k in JOINT_NAMES) and "center" in next(iter(data.values())):
            center = np.array([data[n]["center"] for n in JOINT_NAMES], float)
            sign = np.array([data[n]["sign"] for n in JOINT_NAMES], float)
            return cls(center, sign)
        return cls.from_lerobot(data)

    @classmethod
    def from_lerobot(cls, data: dict) -> "Calibration":
        """Adapt a calibration file written by `lerobot-calibrate`.

        LeRobot stores, per motor, a `homing_offset` such that the joint
        reads mid-range at its zero pose, plus a `drive_mode` flag for
        direction. Both map straight onto our center/sign.
        """
        center = np.full(NUM_JOINTS, float(CENTER_COUNT))
        sign = np.ones(NUM_JOINTS)
        for i, name in enumerate(JOINT_NAMES):
            entry = data.get(name)
            if not isinstance(entry, dict):
                log.warning("calibration missing joint %s; using defaults", name)
                continue
            center[i] = float(CENTER_COUNT - entry.get("homing_offset", 0))
            sign[i] = -1.0 if entry.get("drive_mode", 0) else 1.0
        return cls(center, sign)


def default_lerobot_calibration(robot_id: str, kind: str = "so101_follower") -> Path:
    root = os.environ.get("HF_LEROBOT_HOME") or (Path.home() / ".cache" / "huggingface" / "lerobot")
    return Path(root) / "calibration" / "robots" / kind / f"{robot_id}.json"


class FeetechArm(ArmBackend):
    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        calibration: Calibration | None = None,
        motor_ids: tuple[int, ...] = MOTOR_IDS,
        goal_acceleration: int = 60,   # 0 = instant (harsh), 254 = very smooth
        goal_speed: int = 0,           # 0 = maximum
        torque_limit: int = 800,       # of 1000; leaves headroom before stall
        protocol_end: int = 0,         # STS/SMS little-endian
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.calib = calibration or Calibration()
        self.motor_ids = motor_ids
        self.goal_acceleration = goal_acceleration
        self.goal_speed = goal_speed
        self.torque_limit = torque_limit
        self.protocol_end = protocol_end
        self._port_handler = None
        self._packet_handler = None
        self._sync_read = None
        self._sync_write = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        try:
            import scservo_sdk as scs
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Feetech SDK not installed. Install the hardware extra:\n"
                "    pip install -e '.[robot]'\n"
                "or just the SDK:  pip install feetech-servo-sdk"
            ) from e

        self._port_handler = scs.PortHandler(self.port)
        self._packet_handler = scs.PacketHandler(self.protocol_end)
        if not self._port_handler.openPort():
            raise RuntimeError(f"could not open serial port {self.port!r}")
        if not self._port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"could not set baudrate {self.baudrate}")

        # One sync read covering position+speed+load in a single bus
        # transaction: 3 registers x 6 motors in ~1 ms instead of 18 round
        # trips. At 100 Hz that difference is most of the loop budget.
        self._sync_read = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, ADDR_PRESENT_POSITION, 6
        )
        for mid in self.motor_ids:
            if not self._sync_read.addParam(mid):
                raise RuntimeError(f"sync read: motor {mid} did not respond")
        self._sync_write = scs.GroupSyncWrite(
            self._port_handler, self._packet_handler, ADDR_GOAL_POSITION, 2
        )

        self._connected = True
        self._configure_motors()
        log.info("connected to %d servos on %s @ %d baud", len(self.motor_ids), self.port, self.baudrate)

    def _configure_motors(self) -> None:
        for mid in self.motor_ids:
            self._packet_handler.write1ByteTxRx(self._port_handler, mid, ADDR_GOAL_ACC, self.goal_acceleration)
            self._packet_handler.write2ByteTxRx(self._port_handler, mid, ADDR_GOAL_SPEED, self.goal_speed)
            self._packet_handler.write2ByteTxRx(self._port_handler, mid, ADDR_TORQUE_LIMIT, self.torque_limit)

    def disconnect(self) -> None:
        if self._port_handler is not None:
            try:
                self.set_torque(False)
            finally:
                self._port_handler.closePort()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def set_torque(self, enabled: bool) -> None:
        self._require()
        for mid in self.motor_ids:
            self._packet_handler.write1ByteTxRx(
                self._port_handler, mid, ADDR_TORQUE_ENABLE, 1 if enabled else 0
            )

    def set_torque_limit(self, value: int) -> None:
        self._require()
        value = int(np.clip(value, 0, 1000))
        for mid in self.motor_ids:
            self._packet_handler.write2ByteTxRx(
                self._port_handler, mid, ADDR_TORQUE_LIMIT, value
            )
        self.torque_limit = value

    def _require(self) -> None:
        if not self._connected:
            raise RuntimeError("arm not connected; call connect() first")

    # -- io ----------------------------------------------------------------
    def read(self) -> JointState:
        self._require()
        stamp = time.perf_counter()
        result = self._sync_read.txRxPacket()
        if result != 0:
            raise IOError(f"sync read failed: {self._packet_handler.getTxRxResult(result)}")

        counts = np.empty(NUM_JOINTS)
        speeds = np.empty(NUM_JOINTS)
        for i, mid in enumerate(self.motor_ids):
            counts[i] = self._sync_read.getData(mid, ADDR_PRESENT_POSITION, 2)
            raw_speed = int(self._sync_read.getData(mid, ADDR_PRESENT_SPEED, 2))
            # Speed is sign-magnitude: bit 15 is direction, not two's complement.
            magnitude = raw_speed & 0x7FFF
            speeds[i] = -magnitude if raw_speed & 0x8000 else magnitude

        return JointState(
            q=self.calib.to_rad(counts),
            stamp=stamp,
            dq=self.calib.sign * speeds * RAD_PER_COUNT,
        )

    def write(self, q: np.ndarray) -> None:
        self._require()
        lim = np.vstack([JOINT_LIMITS, np.array([GRIPPER_LIMITS])])
        q = np.clip(np.asarray(q, float), lim[:, 0], lim[:, 1])
        counts = self.calib.to_counts(q)

        self._sync_write.clearParam()
        for mid, c in zip(self.motor_ids, counts):
            self._sync_write.addParam(int(mid), [int(c) & 0xFF, (int(c) >> 8) & 0xFF])
        result = self._sync_write.txPacket()
        if result != 0:
            raise IOError(f"sync write failed: {self._packet_handler.getTxRxResult(result)}")

    def diagnostics(self) -> dict[str, object]:
        self._require()
        temps, volts = [], []
        for mid in self.motor_ids:
            t, _, _ = self._packet_handler.read1ByteTxRx(self._port_handler, mid, ADDR_PRESENT_TEMPERATURE)
            v, _, _ = self._packet_handler.read1ByteTxRx(self._port_handler, mid, ADDR_PRESENT_VOLTAGE)
            temps.append(int(t))
            volts.append(int(v) / 10.0)
        return {"temperature_c": temps, "voltage_v": volts}


def find_ports() -> list[str]:
    """Serial ports that look like a servo bus adapter."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    out = []
    for p in list_ports.comports():
        name = str(p.device)
        if any(k in name for k in ("usbmodem", "ttyACM", "ttyUSB", "usbserial", "COM")):
            out.append(name)
    return sorted(out)
