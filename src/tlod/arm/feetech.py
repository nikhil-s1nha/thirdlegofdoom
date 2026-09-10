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
ADDR_ERROR_STATUS = 65
ADDR_PRESENT_CURRENT = 69

COUNTS_PER_REV = 4096
RAD_PER_COUNT = 2.0 * np.pi / COUNTS_PER_REV
CENTER_COUNT = 2048

# Present_Current (addr 69) is reported in 6.5 mA steps.
AMPS_PER_CURRENT_COUNT = 0.0065

# Goal_Acceleration (addr 41) is an acceleration, in steps/s^2 / 100 --
# equivalently 8.7 deg/s^2 per count at the output shaft, since one
# position step is 0.087 deg. So the register converts to SI as:
RAD_S2_PER_ACC_COUNT = 100.0 * RAD_PER_COUNT   # ~0.1534 rad/s^2 per count

# Bits of the error/status register at addr 65. Undervoltage is the one
# that matters on an undersized supply: the servo latches it, drops
# torque, and the arm goes limp until the rail recovers.
ERROR_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "voltage"),
    (0x02, "angle"),
    (0x04, "overheat"),
    (0x08, "overelectric"),
    (0x20, "overload"),
)


def acc_counts(rad_s2: float) -> int:
    """Convert an acceleration limit to a Goal_Acceleration register value.

    Register 41 holds the *magnitude* of the servo's internal trapezoidal
    ramp, so a bigger number is a harsher start, and 0 disables the ramp
    altogether and gives maximum acceleration. It does not mean "how much
    smoothing" -- 0 and 254 are the two harshest settings, not opposite
    ends of a smoothness scale, and the gentlest useful values are small
    and non-zero.

    This is easy to get backwards, and getting it backwards is expensive:
    winding the register to 254 to "smooth" a brownout in fact asks for
    ~39 rad/s^2, four times the ~9 rad/s^2 that the value of 60 it replaced
    was asking for, and so roughly four times the accelerating current at
    the start of every move. The unit conversion here exists so that
    callers state an acceleration in rad/s^2 and never have to hold the
    polarity in their heads.
    """
    return int(np.clip(round(rad_s2 / RAD_S2_PER_ACC_COUNT), 1, 254))


def decode_errors(status: int) -> list[str]:
    return [name for bit, name in ERROR_BITS if status & bit]

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
                 for n, c, s in zip(JOINT_NAMES, self.center, self.sign, strict=True)},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> Calibration:
        data = json.loads(Path(path).read_text())
        if all(k in data for k in JOINT_NAMES) and "center" in next(iter(data.values())):
            center = np.array([data[n]["center"] for n in JOINT_NAMES], float)
            sign = np.array([data[n]["sign"] for n in JOINT_NAMES], float)
            return cls(center, sign)
        return cls.from_lerobot(data)

    @classmethod
    def from_lerobot(cls, data: dict) -> Calibration:
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
        # Acceleration of the servo's own ramp. Higher is harsher; 0 turns
        # the ramp off entirely, which is harsher still. See acc_counts().
        # 60 counts is ~9.2 rad/s^2, a moderate ramp that keeps the current
        # step at the start of a move well away from the supply's limit.
        goal_acceleration: int = 60,
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
        #
        # This briefly spanned 56..70 to pick up Present_Current, on the
        # theory that current is not clamped by Torque_Limit where load is
        # and so might see a contact that load cannot. Measured across
        # nothing / a book / a hand it separated them by 0.006 A -- one
        # 6.5 mA quantisation step, smaller than the jitter within a single
        # run. It bought nothing and cost fifteen bytes per servo per tick
        # instead of six, on a half-duplex bus that already drops the
        # occasional transaction under load. `probe()` still reads current
        # off the control path, where the cost does not matter.
        self._sync_read = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, ADDR_PRESENT_POSITION, 6
        )
        for mid in self.motor_ids:
            # Local bookkeeping, not a bus transaction: addParam appends
            # the id to the group and returns False only if it is already
            # there. This used to raise "motor {mid} did not respond",
            # which is the one thing it cannot possibly mean.
            if not self._sync_read.addParam(mid):
                raise RuntimeError(f"sync read: motor {mid} registered twice")
        self._sync_write = scs.GroupSyncWrite(
            self._port_handler, self._packet_handler, ADDR_GOAL_POSITION, 2
        )

        # Nothing above this line has spoken to a servo. Opening the port
        # and setting the baud rate are local to the adapter, and the
        # adapter is powered from USB -- so all of it succeeds with the
        # 12 V supply switched off and the arm completely dead.
        #
        # That mattered because what came next was `_configure_motors()`,
        # which ends in `set_torque(True)`, and every write in it discards
        # its return value. So connect() would energise six servos it had
        # never heard from, log "connected to 6 servos" -- a count of the
        # configured id list, not of anything that answered -- and hand
        # back an arm whose first real transaction was the sync read in
        # ArmController.start(). The traceback therefore landed in read(),
        # one frame and one layer away from the actual fault, every time.
        #
        # So: ask first, and refuse to energise a bus that did not answer.
        # Writing goal positions and torque-enable into servos that cannot
        # be read is not a diagnostic inconvenience, it is the setup for a
        # lurch.
        answered, faults = self._survey()
        missing = [m for m in self.motor_ids if m not in answered]
        if missing or faults:
            self._port_handler.closePort()
            raise OSError(self._cannot_connect(missing, faults))

        self._connected = True
        self._configure_motors()
        log.info("connected to %d servos on %s @ %d baud (all answered)",
                 len(answered), self.port, self.baudrate)

    def _survey(self) -> tuple[list[int], list[str]]:
        """Which servos answer, and what any of them calls wrong.

        One round trip each, on a path that runs once per session, so the
        cost does not matter and the answer is worth having before
        anything is energised. Error status is the register to ask for
        because it costs the same as any other and carries the diagnosis
        with it.
        """
        answered, faults = [], []
        for mid in self.motor_ids:
            try:
                err, comm, _ = self._packet_handler.read1ByteTxRx(
                    self._port_handler, mid, ADDR_ERROR_STATUS)
            except Exception:
                continue
            if comm != 0:
                continue
            answered.append(mid)
            named = [name for bit, name in ERROR_BITS if int(err) & bit]
            if named:
                faults.append(f"{mid}:{'+'.join(named)}")
        return answered, faults

    def _cannot_connect(self, missing: list[int], faults: list[str]) -> str:
        """Say which servo is wrong and what to do, in that order."""
        lines = []
        if missing == list(self.motor_ids):
            lines.append(
                f"no servo on {self.port} answered. The adapter is powered "
                "from USB and answered fine, so this is the arm's own 12 V "
                "rail, not the cable to the Pi.")
            lines.append("  1. the inline 12 V switch -- is it on?")
            lines.append("  2. the barrel jack at the adapter, and the supply "
                         "itself (some kits ship 2 A; this needs 5 A)")
            lines.append("  3. a servo that latched a fault holds it until it "
                         "is power cycled -- switch off, count to five, on")
            lines.append(f"  4. `tlod ports` -- {self.port} may not be the arm "
                         "any more; the index moves across replugs")
        elif missing:
            lines.append(
                "servo " + ", ".join(str(m) for m in missing)
                + " did not answer; " + ", ".join(str(m) for m in
                                                  self.motor_ids
                                                  if m not in missing)
                + " did. A gap that starts partway along is the daisy chain: "
                  "check the cable into the first silent one. A servo that "
                  "latched a fault stays silent until it is power cycled.")
        if faults:
            lines.append(
                "faults latched: " + ", ".join(faults)
                + ". These hold until the servo is power cycled -- switch the "
                  "12 V off, count to five, and switch it back on. Undervoltage "
                  "on more than one servo at once is the supply sagging, not "
                  "six coincidences.")
        lines.append("`tlod probe` reads the arm with torque off and is the "
                     "safe thing to run next.")
        return "cannot connect: " + "\n  ".join(lines)

    def _configure_motors(self) -> None:
        for mid in self.motor_ids:
            self._packet_handler.write1ByteTxRx(self._port_handler, mid, ADDR_GOAL_ACC, self.goal_acceleration)
            self._packet_handler.write2ByteTxRx(self._port_handler, mid, ADDR_GOAL_SPEED, self.goal_speed)
            self._packet_handler.write2ByteTxRx(self._port_handler, mid, ADDR_TORQUE_LIMIT, self.torque_limit)
        # disconnect() always leaves torque OFF (see below) so a limp arm is
        # never a surprise between runs. connect() must be the symmetric
        # counterpart and always leave it ON, or a command session started
        # right after e.g. `tlod probe` silently drives goal positions into
        # servos that never move.
        self.set_torque(True)

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
    def read(self, retries: int = 3, backoff: float = 0.001) -> JointState:
        self._require()
        stamp = time.perf_counter()
        # Half-duplex bus: a burst of writes (e.g. the interpolated steps of
        # a goto) can leave stale/echoed bytes sitting in the input buffer.
        # A sync read issued right after picks those up first and its
        # packet parser desyncs -- deterministically, not as random noise,
        # which is why waiting longer between retries alone never helped.
        # Flushing before each attempt discards that leftover backlog so
        # the read starts clean.
        #
        # Which is also why the delay between attempts is short and flat.
        # It was an escalating 10/20/30/40/50 ms, from before the flush was
        # understood to be the actual fix, and that is up to 150 ms spent
        # inside ArmController's lock -- on a thread that is usually the
        # telemetry poller, not the control loop, but holding the lock the
        # control loop needs to issue its next command. Fifteen control
        # ticks would go missing, the following tick would arrive with a
        # correspondingly huge dt, and the arm would lurch. A failing read
        # during motion could therefore cause the jitter it was diagnosing.
        # One millisecond is comfortably longer than a transaction at
        # 1 Mbaud, which is all the flush needs to have something to flush.
        for attempt in range(retries + 1):
            self._port_handler.clearPort()
            result = self._sync_read.txRxPacket()
            if result == 0:
                break
            if attempt == retries:
                raise OSError(
                    f"sync read failed: {self._packet_handler.getTxRxResult(result)}"
                    f"{self._why_silent()}")
            time.sleep(backoff)

        counts = np.empty(NUM_JOINTS)
        speeds = np.empty(NUM_JOINTS)
        loads = np.empty(NUM_JOINTS)
        for i, mid in enumerate(self.motor_ids):
            counts[i] = self._sync_read.getData(mid, ADDR_PRESENT_POSITION, 2)
            # Speed and load are both sign-magnitude: bit 15 is direction,
            # not two's complement. Reading either as signed gives nonsense
            # in one direction only, which is a confusing way to find out.
            raw_speed = int(self._sync_read.getData(mid, ADDR_PRESENT_SPEED, 2))
            speeds[i] = -(raw_speed & 0x7FFF) if raw_speed & 0x8000 else (raw_speed & 0x7FFF)
            raw_load = int(self._sync_read.getData(mid, ADDR_PRESENT_LOAD, 2))
            magnitude = (raw_load & 0x3FF) / 1000.0
            loads[i] = -magnitude if raw_load & 0x400 else magnitude

        return JointState(
            q=self.calib.to_rad(counts),
            stamp=stamp,
            dq=self.calib.sign * speeds * RAD_PER_COUNT,
            load=loads,
        )

    def _why_silent(self) -> str:
        """Which servo went quiet, and what it says is wrong. Best effort.

        "There is no status packet" names nothing: not which of the six
        stopped answering, nor why. A servo that latches a fault -- over
        temperature, over current, undervoltage -- drops off the bus and
        stays off until it is power cycled, and that is indistinguishable
        from a wiring fault or a busy bus unless somebody asks.

        So on the last retry, ask each one individually. It costs six
        round trips on a path that has already failed and is about to
        raise, and it turns an unactionable message into a diagnosis.
        Wrapped completely: this runs while something is already wrong,
        and a failure to explain a failure must not replace it.
        """
        try:
            silent, faults = [], []
            for mid in self.motor_ids:
                err, comm, _ = self._packet_handler.read1ByteTxRx(
                    self._port_handler, mid, ADDR_ERROR_STATUS)
                if comm != 0:
                    silent.append(mid)
                    continue
                named = [name for bit, name in ERROR_BITS if int(err) & bit]
                if named:
                    faults.append(f"{mid}:{'+'.join(named)}")
            parts = []
            if silent:
                parts.append("no reply from servo "
                             + ", ".join(str(m) for m in silent))
            if faults:
                parts.append("faults " + ", ".join(faults))
            if not parts:
                return (" (every servo answers individually and none reports a "
                        "fault, so this is a busy or noisy line rather than a "
                        "dead servo)")
            return (" -- " + "; ".join(parts)
                    + ". A latched fault holds until the servo is power cycled.")
        except Exception:
            return ""

    def write(self, q: np.ndarray) -> None:
        self._require()
        lim = np.vstack([JOINT_LIMITS, np.array([GRIPPER_LIMITS])])
        q = np.clip(np.asarray(q, float), lim[:, 0], lim[:, 1])
        counts = self.calib.to_counts(q)

        self._sync_write.clearParam()
        # strict=True: a length mismatch here would silently command only
        # some of the servos and leave the rest holding their previous
        # goal, which on a moving arm is a half-executed motion rather
        # than an obvious failure.
        for mid, c in zip(self.motor_ids, counts, strict=True):
            self._sync_write.addParam(int(mid), [int(c) & 0xFF, (int(c) >> 8) & 0xFF])
        result = self._sync_write.txPacket()
        if result != 0:
            raise OSError(f"sync write failed: {self._packet_handler.getTxRxResult(result)}")

    def diagnostics(self) -> dict[str, object]:
        """Health of every servo: temperature, rail voltage, current, faults.

        Current and error status are read here rather than in `read()` on
        purpose. `read()` is on the control path and pays for exactly one
        bus transaction; these are six round trips per register and belong
        to whoever is willing to wait for them -- the power diagnostic, a
        telemetry publisher, a health check between moves.

        Each servo reports the voltage at its own terminals, so the spread
        across the six is itself the measurement: a supply that is fine at
        the plug and low at the far end of the daisy chain is a wiring
        problem, and one that is uniformly low is the supply.
        """
        self._require()
        temps, volts, currents, faults = [], [], [], []
        for mid in self.motor_ids:
            t, _, _ = self._packet_handler.read1ByteTxRx(self._port_handler, mid, ADDR_PRESENT_TEMPERATURE)
            v, _, _ = self._packet_handler.read1ByteTxRx(self._port_handler, mid, ADDR_PRESENT_VOLTAGE)
            i, _, _ = self._packet_handler.read2ByteTxRx(self._port_handler, mid, ADDR_PRESENT_CURRENT)
            e, _, _ = self._packet_handler.read1ByteTxRx(self._port_handler, mid, ADDR_ERROR_STATUS)
            temps.append(int(t))
            volts.append(int(v) / 10.0)
            currents.append(int(i) * AMPS_PER_CURRENT_COUNT)
            faults.append(decode_errors(int(e)))
        return {
            "temperature_c": temps,
            "voltage_v": volts,
            "current_a": currents,
            "total_current_a": float(sum(currents)),
            "min_voltage_v": min(volts) if volts else 0.0,
            "faults": faults,
        }


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
