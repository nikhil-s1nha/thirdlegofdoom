# Power, brownout, and motion profiles

For the symptom: **one joint at a time is fine, anything coordinated
jitters, stutters, or goes briefly limp.**

That is a well-known failure of this specific arm, and it is almost
always the power supply. This page has the arithmetic, what other people
found, and what this codebase now does about it.

## Measured first, so you know whether the rest applies

On this rig, with the 12 V **5 A** supply the kit's own bill of materials
asks for, it does not happen:

| | |
|---|---|
| peak current, coordinated multi-joint move | 1.81 A |
| planning budget (5 A x 0.75 headroom) | 3.75 A |
| rail voltage under load | no sag |
| latched servo faults | none |

So the arm has roughly twice the headroom it needs, and the governor is
off by default because there is nothing to govern. `tlod power` prints
your arm's own version of that table — run it before assuming any of the
rest of this page is about you.

Two things still make the page worth keeping. Some kits ship a 12 V 2 A
brick, and Seeed's own spec sheet says 2 A; **that** is the supply that
browns out, and the arithmetic below says why. And the symptom has other
causes that look identical from the outside — a joint fighting a bad
calibration (section 2), and two software bugs that were real here
(section 6).

We spent a while treating one of those as a brownout, and detuned the
arm's motion limits hard to "fix" it. It cost strike speed and fixed
nothing, because the supply was never the problem. Measure before you
slow anything down.

---

## 1. The arithmetic

Six STS3215 on one 12 V rail. From the ST-3215-C018 spec sheet:

| | |
|---|---|
| running current, no load | 180 mA per servo |
| standby current | 30 mA per servo |
| stall current | 2.7 A per servo |
| stall torque | 30 kg.cm (2.94 N.m) |
| rated torque / current | 10 kg.cm / 900 mA |
| torque constant | 11 kg.cm/A |
| over-current trip | >2 A for 2 s, per servo |

The number that decides everything: **180 mA just to turn**, before the
motor lifts anything. Six of those is 1.08 A. So

```
one joint moving,  five holding    ~0.9 A     comfortable on a 2 A supply
five joints moving                 ~1.6 A     before any acceleration
five joints moving, accelerating   ~2.0 A     over a 2 A supply
```

`tlod power` prints your arm's actual version of this table.

That is the entire "works on one motor, fails on complicated movements"
symptom, and it is why it looks like a control bug. It is not one. The
current is roughly linear in the *number of joints moving*, which is the
one variable that changes between the two cases.

### Why it looks like jitter rather than a clean shutdown

It is a loop, not a single event:

1. Several joints accelerate; current demand spikes.
2. The rail sags. Cheap 2 A bricks regulate far slower than the tens of
   milliseconds this takes.
3. The servos' inner loops see less voltage, fall behind their goals, and
   demand more duty — so current climbs further.
4. Below its undervoltage threshold a servo latches a fault and drops
   torque. That joint goes limp and falls.
5. The load disappears, the rail recovers, torque comes back, and the arm
   snaps toward the goal it drifted away from.
6. Which is another current spike. Go to 1.

Meanwhile the sag corrupts serial traffic on the same harness, so bus
reads start failing *exactly* when the arm is moving most — which is why
read failures and jitter show up together and each looks like it might be
causing the other.

### The transient is worse than the average

A position-controlled servo handed a step in position error answers with a
step in PWM duty. The winding is about 1 Ω, so before back-EMF builds,
12 V across it is an inrush on the order of ten amps, bounded by the
torque limit register and inductance rather than by anything in software.
Six of those inside one tick is a collapse that no average-current
calculation would have predicted.

This matters because it means **you cannot fix this by budgeting average
current alone.** You have to stop the step existing.

---

## 2. What other people found

- Seeed specs the SO-ARM101 follower at **12 V 2 A**, and that is the
  supply that browns out. The community answer is a **12 V 5 A** brick
  (5.5×2.1 mm, centre positive). See
  [andymai/so-arm101](https://github.com/andymai/so-arm101), a build
  runbook that hit precisely this and documents the fix.
- It is a recurring upstream issue:
  [huggingface/lerobot#3131](https://github.com/huggingface/lerobot/issues/3131).
- The same runbook's `set-protection` command **caps acceleration** to
  limit current spikes without weakening holding torque — the same lever
  as `arm.servo_accel` here.
- One of its other findings is worth checking before blaming the supply:
  a joint calibrated 111° out of sync drove that motor near stall on every
  cycle, and *that* extra current caused the brownout. Fixing the sync
  fixed the power problem. Run `tlod probe --real` and `tlod first-light`
  before concluding anything.
- Generic servo advice applies too: common ground between the driver board
  and supply, thick short power leads, and never distributing servo power
  through a breadboard.

### On motion profiling generally

Jerk-limited (S-curve) profiling to reduce peak torque and current is
standard practice in motion control, not an invention here. The two names
worth knowing if this needs to grow:

- **[ruckig](https://github.com/pantor/ruckig)** — online jerk-limited
  trajectory generation. Time-optimal, handles a target that moves
  mid-motion, microseconds per cycle. A better solver than the one in
  `tlod/arm/profile.py`; not used only because it is a compiled dependency
  on a board where that is not free.
- **[toppra](https://github.com/hungpham2511/toppra)** — time-optimal path
  parameterisation subject to torque limits. The offline, whole-path
  version of what the governor does per tick.

The specific trick the governor uses — scaling a trajectory in time to fit
an actuator constraint — is classical time-scaling: slow by `s` and
velocity scales by `s`, acceleration by `s²`, so dynamic torque falls as
the square. A 30% slower move costs about half the accelerating current.

---

## 3. The knob that is easy to get backwards

`Goal_Acceleration`, register 41, is an **acceleration magnitude**:

```
value 0     ramp disabled, maximum acceleration   (harshest)
value 1     ~0.15 rad/s^2                         (gentlest)
value 60    ~9.2 rad/s^2                          (the old default)
value 254   ~39 rad/s^2                           (harshest finite)
```

One count is 100 encoder steps/s², and a step is 0.087°, so a count is
8.7 deg/s² at the output shaft.

It is **not** a smoothness dial. 0 and 254 are the two harshest settings,
not opposite ends of a scale. Winding it to 254 to smooth out a brownout
in fact asks for about four times the acceleration of the default 60, and
so roughly four times the current spike at the start of every move.

This is why the config now takes `arm.servo_accel` in **rad/s²** rather
than the raw register value. `tlod.arm.feetech.acc_counts` does the
conversion and refuses to emit 0.

---

## 4. What the code does now

Four layers, outermost first.

### Host-side motion profile — `tlod/arm/profile.py`

Every command, from every path, is shaped before it reaches the servos.
Previously the only bound was a velocity clamp:

```python
step_cap = max_speed * dt
cmd = prev + clip(target - prev, -step_cap, step_cap)
```

which lets commanded velocity go 0 → max in a single tick. That is an
unbounded acceleration and an unbounded jerk in the setpoint stream, which
is exactly the step the servo answers with a current spike.

Now three limits, applied jointly:

| | |
|---|---|
| `safety.max_speed` | rad/s, as before |
| `safety.max_accel` | rad/s², bounds motor torque and so current |
| `safety.max_jerk` | rad/s³, stops that current arriving as a step |

and the limits are **synchronised** across joints: each gets a share of
the budget proportional to its own travel, so they start, peak and arrive
together. That keeps the joint-space path straight and means only the
longest-travelling joint ever runs at the full limit. Measured on a
five-joint move, it cuts peak simultaneous acceleration by about 2.3×
against per-joint limits, and by far more against the old rate clamp.

What the limits cost in speed, measured on the real arm at the values
`configs/real_arm.yaml` now carries: an 80 mm strike drop takes 0.23 s at
best and 0.27 s at an accuracy worth having, and a 150 mm move will not
go below 0.48 s. That is the exchange rate. Detuning these to chase a
brownout that is not happening buys nothing and spends all of it.

### Servo-side ramp — `arm.servo_accel`

The servo's own trapezoidal ramp, discussed above. Belt and braces with
the host profile: the host bounds what is asked for, this bounds how
abruptly the servo chases whatever it is given.

### Power governor — `tlod/arm/power.py`

Off by default; `power.governor: true` turns it on. Two inputs:

- **Feedforward.** Gravity torque and effective inertia at the current
  configuration, from the URDF link masses, converted to predicted current
  through the servo's torque constant. If the configured limits do not fit
  the supply's budget, the whole trajectory is time-scaled until they do.
- **Feedback.** Each servo reports the voltage at its own terminals. If
  the rail sags, derate further, regardless of what the model predicted.

It cuts quickly (0.1 s) and recovers slowly (3 s), because a supply that
has just recovered is the one that will sag again.

The model is deliberately a budget, not a dynamics engine: no off-diagonal
inertial coupling, no Coriolis. It is fitted to the servo's stall point
and runs about 20% high at the rated point, because the spec sheet's own
three operating points are not mutually consistent to better than that.

### A bulk capacitor

The only one of these that acts on the timescale the inrush actually
occupies. 2200–4700 µF, low-ESR, ≥16 V, across V+ and GND as close to the
first servo as it will fit. It supplies the transient the 2 A brick
cannot, and it costs about a pound. Mind the polarity and expect a spark
on connection.

---

## 5. What to actually do

```bash
# 1. Confirm it. Don't infer it.
tlod power -c configs/real_arm.yaml
```

It runs the same move one joint at a time and then all together, and
reports peak current, lowest rail voltage, and any latched servo faults.
If multi-joint sags and single-joint does not, that is the supply. If
neither sags — 1.81 A peak against a 3.75 A budget, as here — the supply
is not your problem and nothing in section 4 will help; go to step 3 and
then to section 6.

**2. Read the label on the brick.** If it says 2 A, replace it: 12 V 5 A,
5.5x2.1 mm, centre positive. Everything else on this page makes a 2 A
supply usable; none of it makes it correct, and per the model even 3 A
clears the problem outright.

```bash
# 3. Check no joint is fighting a bad calibration
tlod probe --real
tlod first-light
```

A joint 100° out of sync draws near-stall current continuously and will
brown out a supply that is otherwise fine.

4. Thick, short power leads and a common ground, always. The bulk
   capacitor only earns its place on an undersized supply; this rig has
   never needed one.

5. `configs/real_arm.yaml` describes a healthy 5 A rig — full motion
   limits, governor off, `supply_current: 5.0`. If you are stuck on 2 A,
   the levers in order of effect are `safety.max_accel`, then
   `arm.servo_accel`, then `power.governor: true` with `supply_current`
   set honestly to what the brick really is. Do not reach for them
   otherwise: they are all speed, and speed is the whole game here.

---

## 6. Two software bugs that made this worse

Both fixed, both worth knowing about because either alone produces jitter
that looks like a power problem.

**Read retries stalled the control thread.** `FeetechArm.read` backed off
10, 20, 30, 40, 50 ms between attempts — up to 150 ms — and did it while
holding `ArmController`'s lock. The telemetry publisher polls state at
20 Hz on its own thread, so a failed read there blocked the control loop
for fifteen ticks. Reads fail most during motion, which is when the
brownout is happening, so this amplified it. The backoff is now a flat
1 ms; the flush before each attempt was always the actual fix.

**A late tick authorised a bigger step.** Every rate limit is `limit * dt`,
and `dt` is measured wall time. A tick that took 200 ms instead of 10 ms
therefore permitted twenty times the motion — a lurch, triggered precisely
when something had already gone wrong. `safety.max_tick_dt` now caps it.
