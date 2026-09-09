"""What a motion will cost the power supply.

The problem this exists for
---------------------------
An SO-101 follower is six STS3215 servos on one 12 V rail. The kit calls
for 5 A; a lot of them ship with 2 A, and Seeed's own spec for the follower
says 2 A. The arm then works perfectly for single-joint moves and falls
apart on coordinated ones, which reads like a control bug and is not one.

The arithmetic, from the ST-3215-C018 spec sheet:

    running current, no load    180 mA   per servo, at 12 V
    stall current               2.7 A    per servo, at 12 V
    stall torque                30 kg.cm = 2.94 N.m at the output shaft
    torque constant             11 kg.cm/A = 1.08 N.m/A

Six servos merely *turning*, carrying nothing, is 6 x 180 mA = 1.08 A. On a
2 A supply that leaves under a third of the budget for the work of holding
the arm up and accelerating it. One joint moving while five hold still is
comfortably inside it; five joints moving at once is not. That is the whole
of the "fine on one motor, jitters on anything complicated" symptom.

What happens next is a loop, not a single event. The rail sags, the servos'
inner loops see less voltage and fall behind their goals, position error
grows, the loops demand more duty, current climbs, and the rail sags
further. Meanwhile the sag corrupts serial traffic on the same harness, so
reads start failing exactly when the arm is moving most.

So the model
------------
Per joint, at a given configuration and commanded acceleration:

    torque  =  gravity(q)  +  M(q)_ii * accel
    current =  quiescent + friction + |torque| / (torque per amp)

and the supply has to carry the sum. Gravity comes from the URDF link
masses and centres of mass; `M(q)_ii` is the exact diagonal of the mass
matrix -- the effective inertia each motor sees about its own axis,
including every link outboard of it.

Deliberately omitted: off-diagonal inertial coupling, Coriolis and
centrifugal terms, and gearbox friction under load. At the speeds this arm
runs, gravity and the diagonal inertia dominate and the rest is noise
against a spec sheet's own tolerance. This is a *budget*, and its job is to
be right about which motions are expensive, not to be a dynamics engine.
Anything relying on it having absolute accuracy should be reading
`Present_Current` off the servos instead, which `tlod power` does.

What this model cannot see, and why that matters
------------------------------------------------
It predicts the current a motion *sustains*. It says nothing about the
current a motion *starts* with, and on a marginal supply that is the one
that trips the brownout.

A position-controlled servo handed a step in position error answers with a
step in PWM duty. The motor winding is about 1 ohm, so for the millisecond
before back-EMF builds, 12 V across it is an inrush of the order of ten
amps, bounded by the torque limit register and the winding's inductance
rather than by anything in this file. Six of those inside the same tick is
a collapse no amount of average-current budgeting would have predicted,
because on a time-average the same motion is unremarkable.

Which is why the budget here is the *second* line of defence and not the
first. The first is not commanding the step at all: bounding the jerk of
the setpoint stream (tlod.arm.profile) and the servo's own internal
acceleration ramp (Goal_Acceleration, see tlod.arm.feetech) is what stops
the transient existing. The third is a bulk capacitor across the rail,
which is the only one of the three that acts on the microsecond timescale
the inrush actually occupies. See docs/power.md.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from tlod.arm import model

log = logging.getLogger(__name__)

GRAVITY = np.array([0.0, 0.0, -9.80665])

# Mass properties from assets/so101_new_calib.urdf, one entry per revolute
# joint in ARM_JOINTS order, expressed in that joint's child link frame --
# which is exactly what model.fk_all() returns, because every joint in this
# URDF already rotates about its own local +z and so needed no reframing.
#
# `com` and `inertia` are the URDF <inertial> origin and tensor; the tensor
# is about the centre of mass, so the parallel-axis term has to be added
# separately (see effective_inertia).
#
# The gripper entry folds in the moving jaw (12 g) at its closed position,
# rather than tracking it as a seventh body: it is 2 cm from the gripper's
# own centre of mass and contributes under 3 mN.m, which is smaller than
# the error in every other number here.
_MASS: tuple[float, ...] = (0.100006, 0.103, 0.104, 0.079, 0.099)
_COM: np.ndarray = np.array([
    [-0.0307604, -1.66727e-05, -0.0252713],   # shoulder_link
    [-0.0898471, -0.00838224,   0.0184089],   # upper_arm_link
    [-0.0980701,  0.00324376,   0.0182831],   # lower_arm_link
    [-0.000103312, -0.0386143,  0.0281156],   # wrist_link
    [0.002445315,  0.000157801, -0.028609724],  # gripper_link + moving jaw
])
_INERTIA: np.ndarray = np.array([
    # ixx, ixy, ixz, iyy, iyz, izz
    [8.3759e-05, 7.55525e-08, -1.16342e-06, 8.10403e-05, 1.54663e-07, 2.39783e-05],
    [4.08002e-05, -1.97819e-05, -4.03016e-08, 1.47318e-04, 8.97326e-09, 1.42487e-04],
    [2.87438e-05, 7.41152e-06, 1.26409e-06, 1.59844e-04, -4.90188e-08, 1.45290e-04],
    [3.68263e-05, 1.7893e-08, -5.28128e-08, 2.5391e-05, 3.6412e-06, 2.1e-05],
    [4.253691e-05, -1.894373e-07, -5.725086e-07, 6.063625e-05, -1.563838e-07, 3.997640e-05],
])


def _inertia_matrix(row: np.ndarray) -> np.ndarray:
    ixx, ixy, ixz, iyy, iyz, izz = row
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


_INERTIA_M: tuple[np.ndarray, ...] = tuple(_inertia_matrix(r) for r in _INERTIA)


@dataclass(frozen=True, slots=True)
class ServoModel:
    """Electrical model of one STS3215, 12 V variant (ST-3215-C018).

    Every figure is off the manufacturer's spec sheet. They are nominal, so
    treat them as a starting point: `tlod power --calibrate` fits
    `torque_per_amp` and `quiescent_current` to what your arm actually
    draws, which is the only version worth trusting for a tight budget.
    """

    stall_torque: float = 2.942       # N.m  (30 kg.cm at 12 V)
    stall_current: float = 2.7        # A
    no_load_current: float = 0.180    # A, running, unloaded
    quiescent_current: float = 0.030  # A, powered and still
    no_load_speed: float = 4.712      # rad/s (45 rpm)
    nominal_voltage: float = 12.0
    # Speed at which a joint is paying the full no-load running current.
    # Not the no-load speed: behind a 1:345 gearbox most of that 180 mA is
    # gearbox drag and iron loss, which are near constant once turning
    # rather than proportional to how fast. Scaling friction by
    # speed/no_load_speed instead would have five joints creeping at
    # 1 rad/s costing 150 mA between them when they really cost 750 mA,
    # and that three-quarters of an amp is most of what is missing.
    friction_speed: float = 0.5       # rad/s

    # Manufacturer's torque constant, 11 kg.cm/A at the output shaft.
    torque_per_amp: float = 1.0787     # N.m per amp of load current

    def check(self) -> dict[str, float]:
        """The spec sheet's own operating points, as this model sees them.

        Worth keeping honest about: the three points the manufacturer
        gives are not mutually consistent to better than about 20%. Stall
        (2.94 N.m at 2.7 A) and the quoted torque constant agree to 2%,
        but the rated point (0.98 N.m at 900 mA) implies a stiffer motor
        than either. Real motors are not linear and datasheets are not
        measured the same way twice, so the model is fitted to stall --
        the end that matters for a current limit -- and lands about 20%
        high at the rated point.

        Which is the argument for `tlod power --calibrate`: 20% is a lot
        of a 2 A budget, and the arm can measure its own answer.
        """
        return {
            "stall_a": float(self.current(np.array([self.stall_torque]))[0]),
            "rated_a": float(self.current(np.array([0.9807]),
                                          np.array([self.no_load_speed]))[0]),
        }

    def current(self, torque: np.ndarray, speed: np.ndarray | None = None) -> np.ndarray:
        """Per-joint supply current, amps, for a torque and a speed.

        Speed matters only through friction: a servo that is turning pays
        the no-load running current, one that is merely holding pays the
        much smaller quiescent draw. Six servos' worth of that difference
        is nearly a whole amp, so it is not a detail.
        """
        torque = np.abs(np.asarray(torque, float))
        friction = np.full(torque.shape, self.quiescent_current)
        if speed is not None:
            turning = np.clip(np.abs(np.asarray(speed, float)) / self.friction_speed, 0.0, 1.0)
            friction = friction + (self.no_load_current - self.quiescent_current) * turning
        return friction + torque / self.torque_per_amp


@dataclass(frozen=True, slots=True)
class PowerBudget:
    """What the supply can be asked for.

    `limit` is deliberately below the supply's rating. A bench supply's
    number is a continuous rating measured at its own terminals; what
    reaches the servos is that minus the drop down a metre of thin wire and
    five daisy-chain connectors, and the transient the arm asks for lasts
    tens of milliseconds, which is faster than most cheap supplies regulate.
    Budgeting the full rating means running the sag right up to the point
    the model stops being able to see it.
    """

    supply_current: float = 2.0       # A, what the brick is rated for
    headroom: float = 0.75            # fraction of the rating to plan for
    min_voltage: float = 10.5         # V, below which the rail is sagging

    @property
    def limit(self) -> float:
        return self.supply_current * self.headroom


class PowerModel:
    """Predicts joint torque and supply current for the SO-101."""

    def __init__(self, servo: ServoModel | None = None, budget: PowerBudget | None = None) -> None:
        self.servo = servo or ServoModel()
        self.budget = budget or PowerBudget()

    def terms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gravity torque (N.m) and effective inertia (kg.m^2), per joint.

        Both come out of one forward-kinematics pass because both need the
        same joint frames and world centres of mass, and this is called
        from the control loop.

        Gravity, for joint i, is the moment about its axis of the weight of
        every link outboard of it. Inertia is the diagonal of the mass
        matrix: for each of those links, the parallel-axis term from its
        mass at its perpendicular distance from the axis, plus its own
        rotational inertia resolved onto that axis. Off-diagonal coupling --
        one joint's acceleration loading another -- is left out, and is the
        one real approximation here.
        """
        frames = model.fk_all(np.asarray(q, float)[:5])
        world_com = [f[:3, :3] @ _COM[k] + f[:3, 3] for k, f in enumerate(frames[:5])]

        gravity = np.zeros(5)
        inertia = np.zeros(5)
        for i in range(5):
            axis = frames[i][:3, 2]
            origin = frames[i][:3, 3]
            moment = np.zeros(3)
            total = 0.0
            for k in range(i, 5):
                offset = world_com[k] - origin
                moment = moment + np.cross(offset, _MASS[k] * GRAVITY)
                lever = np.cross(axis, offset)
                R = frames[k][:3, :3]
                total += _MASS[k] * float(lever @ lever) + float(axis @ (R @ _INERTIA_M[k] @ R.T) @ axis)
            gravity[i] = -float(axis @ moment)
            inertia[i] = total
        return gravity, inertia

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Torque each joint must hold to keep the arm up, N.m."""
        return self.terms(q)[0]

    def effective_inertia(self, q: np.ndarray) -> np.ndarray:
        """Inertia each motor sees about its own axis, kg.m^2."""
        return self.terms(q)[1]

    def joint_torque(
        self,
        q: np.ndarray,
        accel: np.ndarray | None = None,
        *,
        worst_case: bool = True,
    ) -> np.ndarray:
        """Torque demanded of each motor, N.m.

        `worst_case` adds the gravity and inertial terms as magnitudes
        rather than signed. That is the right answer for a budget and the
        wrong one for physics: accelerating downhill genuinely costs less
        than accelerating uphill, because the inertial torque subtracts
        from the gravity torque. But a limit has to hold for both
        directions of the same move, and it is set before anyone knows
        which way the arm will be asked to go. Sizing to the cheap
        direction would put the expensive one over budget every time.
        """
        gravity, inertia = self.terms(q)
        if accel is None:
            return gravity
        dynamic = inertia * np.asarray(accel, float)[:5]
        if worst_case:
            return np.abs(gravity) + np.abs(dynamic)
        return gravity + dynamic

    def current(
        self,
        q: np.ndarray,
        accel: np.ndarray | None = None,
        speed: np.ndarray | None = None,
        *,
        worst_case: bool = True,
    ) -> np.ndarray:
        """Per-joint supply current, amps. Five arm joints; the gripper is
        not modelled because it carries no part of the arm's weight."""
        torque = self.joint_torque(q, accel, worst_case=worst_case)
        return self.servo.current(torque, None if speed is None else np.asarray(speed)[:5])

    def total_current(self, q, accel=None, speed=None, *, worst_case: bool = True,
                      include_gripper: bool = True) -> float:
        total = float(self.current(q, accel, speed, worst_case=worst_case).sum())
        if include_gripper:
            # The gripper motor is on the same rail whether or not it is
            # doing anything, and forgetting it is a 30-180 mA hole in a
            # 1.5 A budget.
            total += self.servo.quiescent_current
        return total

    def headroom(self, q: np.ndarray, accel=None, speed=None) -> float:
        """Amps of budget left. Negative means the supply is being oversold."""
        return self.budget.limit - self.total_current(q, accel, speed)

    def feasible_scale(
        self,
        q: np.ndarray,
        accel: np.ndarray,
        speed: np.ndarray | None = None,
        floor: float = 0.1,
    ) -> float:
        """Largest time-scale factor whose current fits the budget, in (0, 1].

        Slowing a trajectory by `s` in time scales velocity by `s` and
        acceleration by `s^2`, so both the inertial torque and the friction
        of turning come down with it -- but by different powers, and
        friction saturates. That rules out the closed form and leaves a
        bisection, which is cheap here because the expensive part (one
        forward-kinematics pass for gravity and inertia) happens once
        outside the loop.

        Gravity does not scale at all: holding the arm up costs the same
        however slowly it moves. So there is a floor below which slowing
        down buys nothing, and hitting it means the supply cannot carry
        this pose at any speed. That is a hardware answer, not a trajectory
        one, and the caller is expected to say so rather than crawl.
        """
        gravity, inertia = self.terms(q)
        gravity = np.abs(gravity)
        dynamic = np.abs(inertia * np.asarray(accel, float)[:5])
        speed = np.zeros(5) if speed is None else np.abs(np.asarray(speed, float)[:5])

        def draw(s: float) -> float:
            torque = gravity + dynamic * s * s
            return float(self.servo.current(torque, speed * s).sum()) + self.servo.quiescent_current

        if draw(1.0) <= self.budget.limit:
            return 1.0
        if draw(floor) > self.budget.limit:
            return floor
        lo, hi = floor, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if draw(mid) <= self.budget.limit:
                lo = mid
            else:
                hi = mid
        return lo

    def worst_case_scale(self, q: np.ndarray, limits) -> float:
        """Scale factor that keeps the *limits* affordable at this pose.

        Judged against what the profile is permitted to do, not what it
        happens to be doing. Governing on the instantaneous draw instead
        would be a feedback loop with the arm inside it: the derate would
        only tighten once the arm was already accelerating, which is after
        the current spike it exists to prevent.
        """
        return self.feasible_scale(
            q,
            np.full(5, float(limits.max_accel)),
            np.full(5, float(limits.max_speed)),
        )

    def supports_holding(self, q: np.ndarray) -> bool:
        """Whether the supply can carry this pose standing still.

        If not, no amount of motion profiling helps and the honest answer
        is a bigger supply.
        """
        return self.total_current(q) <= self.budget.limit


class PowerGovernor:
    """Holds the arm inside its supply, and notices when it fails to.

    Two inputs, deliberately of different kinds:

    *Feedforward*, from the model: at this configuration, how much of the
    configured speed and acceleration can the supply afford? This acts
    before the motion, which is the only time acting is any use.

    *Feedback*, from the servos' own voltage reports: is the rail actually
    sagging? This catches everything the model is wrong about -- a supply
    that is worse than its label, a long thin power lead, a stiff joint --
    at the cost of only responding once the sag has happened.

    The two are combined by taking the tighter, and the result is slewed
    rather than applied instantly: down quickly, because a sag needs
    answering now, and back up slowly, because a supply that has just
    recovered is precisely the one that will sag again if the arm
    immediately resumes what caused it. An arm that oscillates between fast
    and slow is worse than one that is honestly slow.
    """

    def __init__(
        self,
        model: PowerModel | None = None,
        *,
        cut_time: float = 0.1,
        recover_time: float = 3.0,
        pose_interval: float = 0.2,
    ) -> None:
        self.model = model or PowerModel()
        self.cut_time = cut_time
        self.recover_time = recover_time
        # How often the pose-dependent term is recomputed. It costs an
        # inertia matrix and a feasibility search -- ~12 ms on a Pi 5,
        # measured, which is more than policy and IK and the servo bus
        # together and does not fit in a control tick. It also does not
        # need to run at control rate: its output is fed through the
        # cut_time/recover_time filter below, which is an order of
        # magnitude slower than a tick, so recomputing every tick buys
        # precision the filter immediately throws away. The voltage term
        # is *not* cached; see update().
        self.pose_interval = pose_interval
        self.scale = 1.0
        self.voltage_scale = 1.0
        self.last_voltage: float | None = None
        self.sag_events = 0
        self._pose_scale = 1.0
        self._since_pose = float("inf")
        self.pose_evaluations = 0

    def note_voltage(self, volts: float) -> None:
        """Feed in a measured bus voltage, from anywhere that has one.

        Below the threshold the rail is already sagging under load, so the
        derate is cut proportionally to how far under it has gone. The
        servos' own undervoltage protection sits some way below this: the
        point is to back off before it fires, because when it fires the
        servo drops torque and the arm falls.
        """
        self.last_voltage = float(volts)
        floor = self.model.budget.min_voltage
        nominal = self.model.servo.nominal_voltage
        if volts >= floor:
            self.voltage_scale = min(1.0, self.voltage_scale + 0.05)
            return
        self.sag_events += 1
        # How far into the sag, as a fraction of the span between the
        # threshold and the point the servos give up entirely.
        depth = (floor - volts) / max(floor - 0.75 * nominal, 0.1)
        self.voltage_scale = float(np.clip(1.0 - depth, 0.1, 1.0))

    def update(self, q: np.ndarray, limits, dt: float) -> float:
        """Advance the governor one tick. Returns the derate to apply.

        Two terms, deliberately on different clocks. The pose term asks
        what this configuration can afford and is expensive, so it runs
        at `pose_interval`; a pose cannot change enough in 200 ms to
        matter to a filter whose own recovery is measured in seconds.
        The voltage term is read every tick, because a sagging rail is
        the one input that must be answered immediately -- it arrives
        from HealthMonitor between pose evaluations, and delaying it is
        exactly the failure the governor exists to prevent.
        """
        self._since_pose += dt
        if self._since_pose >= self.pose_interval:
            self._since_pose = 0.0
            self._pose_scale = self.model.worst_case_scale(q, limits)
            self.pose_evaluations += 1
        target = min(self._pose_scale, self.voltage_scale)
        tau = self.cut_time if target < self.scale else self.recover_time
        alpha = 1.0 if dt >= tau else dt / tau
        self.scale += (target - self.scale) * alpha
        self.scale = float(np.clip(self.scale, 0.05, 1.0))
        return self.scale


class HealthMonitor:
    """Polls the servos for rail voltage and faults, and tells the governor.

    A separate low-rate thread rather than part of the control loop: this
    is two dozen serial round trips, which does not belong in a 10 ms tick
    even though it does belong in the loop's decision-making. Twice a
    second is plenty -- a sag lasts as long as the motion causing it, and
    the governor's own recovery is slower than that anyway.

    Goes through `ArmController.diagnostics()` rather than the backend
    directly, so the reads take the bus lock and cannot interleave with
    the control loop's writes.
    """

    def __init__(self, controller, governor: PowerGovernor, rate_hz: float = 2.0) -> None:
        self.controller = controller
        self.governor = governor
        self.rate_hz = rate_hz
        self.faults: set[str] = set()
        self.peak_current = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="power-health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        period = 1.0 / max(self.rate_hz, 0.1)
        while self._running:
            t0 = time.perf_counter()
            try:
                self._sample()
            except Exception as e:
                # A transient bus failure must not kill the monitor. It
                # would take the governor's only feedback path with it and
                # leave the arm running on the model's prediction alone --
                # silently, which is the objectionable part.
                log.debug("power health sample failed: %s", e)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _sample(self) -> None:
        d = self.controller.diagnostics()
        volts = d.get("min_voltage_v")
        if volts:
            self.governor.note_voltage(float(volts))
        self.peak_current = max(self.peak_current, float(d.get("total_current_a", 0.0)))
        new = {f for per in d.get("faults", []) for f in per} - self.faults
        if new:
            self.faults |= new
            log.warning("servo faults latched: %s (rail %.1f V) -- see docs/power.md",
                        sorted(new), volts or 0.0)
