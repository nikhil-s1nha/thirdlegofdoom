# Did the slap land?

The question a camera cannot answer. At the moment of contact the arm is
between an overhead camera and the contact point, occluding exactly the
thing that needs to be seen, and at 30 fps a frame is 33 ms against an
event that lasts a few milliseconds.

The encoders answer it instead, and this is the record of how — including
the four ways it was wrong first, because each of those is a trap that
looks like a different problem from the outside.

## The rule

The strike commands a floor **below** the hand. So there are only two
endings:

```
    ── 91 mm ── hover            (hand + hover_height)
         │
         ▼
    ── 28 mm ── the hand plane   <- paddle ends up NEAR here if blocked
         ┊   17 mm of band        (press_depth)
    ── 11 mm ── commanded floor  <- paddle reaches here if not
```

**Stopped short of the floor = hit. Reached it = dodge.** Both heights
come off the encoders, so neither is late, neither is filtered, and
neither costs a bus transaction the control loop was not already making.

`tlod play --real` takes no sensor argument. There is one way to judge a
round and it is this one.

## The measured numbers

On an SO-ARM101 at 12 V 5 A, `torque_limit` 350/1000, floor 11 mm, hand
plane 28 mm (`vision.hand_height`, an assumed palm height — one camera
cannot get depth):

| | paddle stops at | short of the floor |
|---|---|---|
| empty table | 6–10 mm | −5 to −1 mm |
| a hand | 17–23 mm | +6 to +12 mm |
| **threshold** | **13.5 mm** | **+2.5 mm** |

Empty rows are `strike_bench` with nothing on the table, `descent
arrived` on every one. Hand rows are a play session. The gap is 10–17 mm,
so the threshold sits at 2.5 mm with ~3.5 mm of room on either side.

The paddle ends up **below** its floor on an empty table. It is being
driven down and gravity is helping; there is nothing to stop it at 11 mm.

## Why the threshold is a percentage of the band

`band_fraction`, 15% of the floor-to-hand distance, with `margin` as an
absolute floor for a band too thin for a fraction to mean anything.

Absolute millimetres have to be re-tuned whenever `press_depth`, the
torque limit, or the arm's load changes, because every one of those moves
where the paddle ends up. "Did it end up nearer the hand or nearer the
floor" does not move.

That is the theory. In practice it was still calibrated wrong once, and
the reason is worth stating plainly:

> **A threshold calibrated against one version of the motion does not
> survive changing the motion.**

Both times this was wrong, that was why. The test in
`TestTheThresholdSitsBetweenTheClusters` therefore asserts the threshold
is *centred* between the measured clusters with 2 mm of clearance each
side — not merely that it classifies the rows correctly. Change the
motion and it fails loudly rather than drifting into a cluster.

## The four wrong answers

Each of these ran on hardware and each looked like something else.

### 1. Torque, during the swing

`ServoLoadContactSensor`. Measured across nothing / a book / a hand, peak
`Present_Load` during the swing: **0.330 / 0.326 / 0.350**. A rigid book —
the easiest thing there is to feel — landed *between* the other two.

Load climbs monotonically to its ceiling in every run, empty table
included, because the arm is accelerating and braking its own mass. The
whole swing is one transient and nothing measured during it is about what
was hit.

### 2. Torque, held still

`ServoPressContactSensor`. The same three, held at the bottom: **0.001 /
0.038 / 0.037**. This works. It costs ~300 ms of six servos stalled at
their torque limit on every strike, and sustained stall current is what a
5 A supply has least of — `press_hold` went to 450 ms one afternoon and
the bus started dropping transactions the same afternoon.

`Present_Current` (addr 69) separated the same three by 0.006 A, one
6.5 mA quantisation step, smaller than the jitter within a single run.

Both classes are still in `game/contact.py`, marked `UNUSED`, carrying
their measurements. Nothing constructs them; a test enforces that.
Deleting them means the next person re-runs the same experiments.

### 3. The camera, mid-descent

`ProximityContactSensor` was the CLI default for several commits after
the encoder sensor landed — the old default outliving the sensor that
replaced it. It compares the tracked hand against the tool, and it
judged on **every tick of the descent**.

The strike is aimed where the hand was at commit, so the horizontal test
starts at ~0 and the verdict turns entirely on the vertical one. That
enters the tolerance band ~105 ms into a ~310 ms strike, against a hand
estimate ~95 ms stale — so the round was decided on where the hand had
been at t≈10 ms. A human sees the motion at ~65 ms and needs another
~150 ms to clear. **The dodge was never in the data.** Every committed
strike scored a hit.

It now waits for the paddle to arrive, gated on `pressing`, latched so
the late frames still count. Tier B still uses it, because a simulated
arm is never blocked by anything and the swept band is all there is to
read.

### 4. Declaring the strike over before the arm stopped

The subtle one, and the one that cost the most.

`Motion._complete` asks `controller.settled()`, which is true once the
**commanded setpoint** stops changing. That is not the arm. From a bench
trace: at the instant the strike called itself done, the paddle was
**16 mm above its floor and travelling at 0.18 m/s**, and it went on
moving for another ~360 ms.

Everything downstream hangs off that instant — `pressing`, and through it
every sensor. So the gap being measured was not a hand, it was travel the
arm had not done yet.

The codebase already knew: `ServoPressContactSensor`'s docstring rejects
`settled()` for exactly this ("it reported settled while the paddle still
had 10 mm to travel") and gates on `pressing` instead — but `pressing` is
set from `_complete()` → `settled()`, so the escape hatch led back to the
thing it was escaping.

**The descent now ends when the arm stops, not when the asking stops.**

## How a round runs

```
COMMIT     aim at the hand, floor = hand − press_depth, arm the sensor
   ↓
DESCEND    min-jerk plan, capped at strike_speed, torque 350/1000
           each tick: the game reads the pose and calls motion.observe(z)
   ↓
END ON     arrived    within ARRIVE_EPSILON (2 mm) of the floor
           stopped    same height for STILL_DWELL (150 ms)
           timed out  settle_timeout backstop -- a bug; it warns
   ↓
PRESS      lean for press_hold
   ↓
READ       after settle:  short = reached − floor
           HIT if short >= band_fraction of the band, else DODGE
```

Three things in there are load-bearing and none are obvious:

**Stopped, not arrived.** On a hit the paddle *never* arrives — it stalls
against the hand. Stopping is what both outcomes have in common and
where it stopped is the answer. Waiting for arrival hangs every hit.

**Arrival short-circuits the dwell.** A clean dodge ends the moment it
lands rather than waiting 150 ms to prove it has stopped, so only a
blocked paddle pays, and a blocked paddle is not going anywhere. Without
this every dodge stalls the servos for an extra 150 ms.

**`observe()` rather than reading the bus.** The motion is fed the height
the caller already read this tick. A second sync read inside a strike is
not free: one failed mid-swing, the control loop answered a failed policy
tick by e-stopping, and the arm froze directly above the hand it was
aiming at.

`STILL_DWELL` is 150 ms because a min-jerk plan decelerates to zero
velocity *by design*, so near the bottom the arm is always crawling
whether it has arrived or not. At 60 ms that reads as stopped and the
descent ends in mid-air — measured: dodges quitting at 17–21 mm instead
of reaching 11 mm, hits at 24–27 mm against a 28 mm hand instead of
pressing in. 150 ms is longer than the recovery creep in the bench trace
(1–2 mm per 26 ms), so a decelerating arm keeps resetting the timer.

## Reading the round line

Printed every round, whichever way it went:

```
paddle stopped 18 mm, floor 11 mm, hand 28 mm -> +7 mm short (needs 3)  [descent stopped]
```

| | |
|---|---|
| `descent arrived` + low | reached the floor. Should be a dodge |
| `descent stopped` + high | parked on something. Should be a hit |
| `descent timed out` | neither gate fired. A bug, not a round; it also warns |
| `no reading` | the press never produced a readable moment — chase this, it is not a dodge |

A verdict alone is unfalsifiable: "dodged" looks identical whether the
paddle stopped on a hand and the threshold was too wide, or the floor was
above the hand so there was nothing to stop short of, or the hand simply
compressed. Each has a different fix. Guessing between them is what
several rounds of this cost.

## The sixth wrong answer, and the one that cost most: the reference

**The paddle never reaches its commanded floor.** Everything above
compares where it stopped against where it was *told* to stop, and on
this rig those are 10 mm apart.

Measured on the bench at 0.36/0.165, commanded floor 20 mm, nothing under
the paddle:

```
settled at 11 mm, 8 mm, 9 mm, 9 mm      -- and held there for 300+ ms
```

Not a transient. The trace shows the command sitting at 21 mm and the arm
resting at 8-9 mm for the rest of the hold. The arm simply cannot hold
itself at its commanded height in that pose.

Why a strike and not a `move`: `goto_pose` re-aims at what the encoders
report and converges to a couple of millimetres. `Strike` deliberately
does not -- it is meant to be fast, and a second pass is a second, slower
descent. So a strike carries the arm's full static sag with nothing
correcting it, and `ArmController.compensate` does not cover it either:
that model was fitted at hover height and the arm is in a very different
pose at the floor.

Now put the hand cluster next to it, same geometry, hand left
deliberately in the way every round:

```
empty table   8-11 mm      a hand   13-16 mm
```

**The signal was there the whole time** -- 2-5 mm, perfectly serviceable.
Both clusters just sit ~10 mm *under* the floor, and the threshold was
placed 1.5 mm *above* it. Above both. So every round scored a dodge,
including the ones that visibly hit a hand, and no amount of moving
`press_depth`, `strike_torque`, `margin` or `band_fraction` could ever
have found it, because all of them adjust the paddle or the threshold and
none of them adjust the reference.

`arm.strike_sag` is that reference: how far below its commanded floor an
unobstructed strike actually rests. `strike_bench` over an empty table
prints the number to put in it. Set it to centre the threshold between
the two clusters rather than at the mean of either -- at 9.5 mm the
resting reference lands at 10.5, an empty strike reads -2.5..+0.5 and a
blocked one +2.5..+5.5 against a 1.5 mm threshold, a millimetre of room
each side.

It is the `band_fraction` rule in another form, and the reason it kept
biting for so long is that the reference was never measured at all --
only assumed, from the one number in the system that is a command rather
than a measurement.

## The fifth wrong answer: scoring a miss as a dodge

Added after a session where the arm went 7-14 in one run and 0-7 in the
next with nothing changed in between, and the round lines gave no way to
tell which had happened.

A strike aimed off the hand reaches its floor. So does a strike the human
dodged. Every height the sensor reads is identical:

```
paddle stopped 18 mm, floor 22 mm, hand 30 mm -> -4 mm short (needs 2)
```

That line is the same whether the hand moved or the paddle came down
60 mm to the side of it, and the two have opposite fixes -- one is
detection working correctly, the other is the aim and no threshold
anywhere repairs it.

The horizontal distance was available the whole time: `poll()` already
receives both the tool position and the tracked hand. It is now on the
round line, and rounds beyond `MISS_RADIUS` (50 mm, half a palm plus a
little) are counted and called out at the end of the run:

```
paddle stopped 18 mm, floor 22 mm, hand 30 mm -> -4 mm short (needs 2)
  [MISSED: came down 61 mm to the side of the hand, ...]
```

Why this rig produces them: `where_is_my_hand --truth 0.25 0.0` reports
the palm at 264-267 mm, a standing ~24 mm bias, and its own residual
analysis says no hand height explains the ray, so it is the extrinsics.
`vision-check` agrees at 24 mm mean 19. A palm is about 90 mm across, so
a 24 mm bias lands the paddle near the edge and sometimes past it --
intermittently, which is the worst way for it to fail.

## How deep to press: shallower, and it is a cliff

`press_depth` reads like a safety margin -- press further under the hand
and the hand is more definitely in the way. It is the opposite. Floor =
the 30 mm hand plane less the press; every row is a hand deliberately
left under the paddle:

| press | floor | where the paddle ended up | verdict |
|---|---|---|---|
| 25 mm | 5 mm | 4 mm | dodge, every round |
| 20 mm | 10 mm | 6-9 mm | dodge, every round |
| 15 mm | 15 mm | 12-16 mm | dodge, every round |
| 8 mm | 22 mm | **held at 24-26 mm** | **hit** |

Not a gentle trend -- a cliff between 15 and 8.

**Why.** These servos are proportional position controllers, so push-down
force goes with position error. A floor 15 mm under the palm leans on it
several times as hard as a floor 3 mm under it. Flesh yields: under some
force the palm holds the paddle up, over it the palm flattens and the
paddle carries on to its floor. Pressing deeper does not put the hand
more firmly in the way, it pushes harder *through* it -- and reaching the
floor is what the sensor calls a dodge. **So pressing harder is how a
genuine hit gets scored as a miss.**

The corollary is that "how thick is a hand" is the wrong question. What
matters is where the paddle *comes to rest* on one, which is a force
balance and not an anatomical fact. On this rig that is 24-26 mm, and it
collapses to the floor once the lean passes about 10 mm.

### The gap is 2 mm, and press_depth cannot widen it

At 8 mm the clusters are +2..+4 mm short for a hand and -0..-15 for an
empty table. The threshold sits at 1 mm, centred, with a millimetre
either side. That works, and it is thin.

`press_depth` only slides the clusters; it cannot pull them apart.
**`arm.strike_torque` can** -- it sets how hard the paddle leans, so it
sets where the palm's resistance balances it. The trend is three points
long and monotone:

- 500, leaning hard: drove through a hand on all twelve strikes of a session
- 350, leaning 3 mm: the same hand held it, +2..+4 mm short
- 350, leaning 15 mm: through again

Downward is untried and is the obvious next experiment. What bounds it
from below is the *swing*, not the press: bench traces peak at 0.21-0.30
of rated torque braking the arm's own mass, so a ceiling under ~0.30
starts clipping the descent and an empty strike stops reaching its floor
-- which kills the other half of the signal. Sweep it over an empty
table and keep the lowest value where `descent arrived` still holds
every time:

```bash
python3 scripts/strike_bench.py 0.36 0.165 --torque 300
python3 scripts/strike_bench.py 0.36 0.165 --torque 250
```

## The two ways the geometry can silently break

- **The floor is at or above the hand.** Then a touched paddle and an
  untouched one stop in the same place and everything reads as a dodge.
  `press_depth` puts the floor below; `safety.min_height` can clamp it
  back up, and `hover_height + press_depth > max_drop` raises it too.
  The tell is a floor and a hand at the same height in the round line.
- **The hand compresses to the floor.** Flesh is soft and the paddle
  keeps pushing for `press_hold`. If it squashes the last millimetres out
  of a palm, the band closes.

## Tuning it on your own rig

```bash
python3 scripts/strike_bench.py 0.19 0.10     # empty table, 4-5 strikes
```

Read `settled at N mm` — that is the empty-table cluster. Then play a
session and read the round lines for the hand cluster. Put the threshold
between them:

```bash
tlod play --real --contact-band 0.15
```

Lower it if real hits score as dodges, raise it if the empty table scores
as hits. The end-of-run line prints the peak shortfall the session saw.

Two things that void a bench run: a `SHORT of 91` hover warning (a short
hover is a short drop, the variable being controlled for), and running at
a different x,y from where the game plays.

The bench builds its controller with `arm.flex_gain` / `arm.flex_offset`,
the same as the game. It has to: without them it commands and reports in
the kinematic frame while `tlod play` works in the real one, and at full
reach those are 33 mm apart -- so the empty-table cluster measured here
and the hand cluster measured in a play session would not be comparable
at all. That is the same mixed-frame bug `Strike` and `Feint` were fixed
for, which the bench kept for a while afterwards.

## Known loose end

The hover comes up ~13 mm short of what it asks for. IK reaches the
requested height fine, so it is the arm sagging under gravity rather than
a reach limit. It has been true across every run including the good ones,
so it is not implicated in detection — but it means a strike drops ~67 mm
where the geometry says 80 mm.
