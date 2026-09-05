# Why the robot should be the one slapping

## The finding

Making the robot the **slapper** and the human the **dodger** removes
latency from the critical path. Not by making anything faster — by
changing who is responding to whom.

Latency only taxes the responder. When the robot initiates, its ~250 ms
sense-to-motion pipeline is spent *before* the strike begins, where nobody
is waiting on it. The human's clock starts at contact-minus-strike-time.

A second effect compounds it: a hand *waiting to be slapped* is nearly
stationary, so a 300 ms-old position estimate is still a correct estimate.
Kalman prediction stops being load-bearing for aiming. (It stays useful
for the case where the human is already withdrawing mid-strike.)

## Measured strike performance

Simulated, using datasheet-derived servo parameters. **These need M6
validation on real hardware.** Tip velocity is Jacobian-derived, not
finite-differenced — an early version of this measurement differenced
position over sub-millisecond intervals and produced impossible readings
of 7-15 m/s.

Strike geometry: tool hovers directly above the hand and drives straight
down. Impact energy assumes ~0.15 kg effective tip mass.

| drop | 2.0 rad/s | 3.5 rad/s | 5.0 rad/s |
|---|---|---|---|
| 5 cm | 222 ms / 0.41 m/s | **175 ms / 0.53 m/s** | 137 ms / 0.67 m/s |
| 8 cm | 280 ms / 0.54 m/s | **210 ms / 0.70 m/s** | 161 ms / 0.89 m/s |
| 10 cm | 315 ms / 0.58 m/s | 228 ms / 0.81 m/s | 182 ms / 1.01 m/s |
| 15 cm | 357 ms / 0.61 m/s | 258 ms / 1.05 m/s | 190 ms / 1.35 m/s |

Human escape budget, from the literature:

| | |
|---|---|
| simple visual reaction, expecting it | 150–250 ms |
| hand withdrawal once initiated | 80–150 ms |
| **total** | **230–400 ms** |

## The design rule: hover close, strike short

Short strikes are better on both axes simultaneously — faster to land
*and* softer on impact. This is unusual and worth exploiting. A big
windup is worse in every respect: slower, harder-hitting, and more
visible to the human, which starts their reaction clock earlier.

Target: **8 cm drop, ~210 ms, ~0.7 m/s**. For scale, a casual human
high-five is 1–3 m/s, so this taps noticeably more softly than a person.

The margin against a fast human is genuinely thin, which makes for a
better game than a robot that always wins. Difficulty should be tuned by
adjusting hover distance and reaction delay, not by crippling the arm.

## What this does to the vision requirements

Requirements collapse in the place they were hardest, and the difficulty
moves somewhere much more tractable.

| need | robot dodges | robot slaps |
|---|---|---|
| aiming | 60 fps, <20 ms, prediction critical | static target, 30 fps, latency irrelevant |
| presence / safety gating | — | needed, low rate |
| hit vs miss | vision | **vision cannot do this** |
| when to strike | — | **the new hard problem** |

**Hit detection is the catch.** At the moment of contact the arm is
directly between the camera and the contact point, occluding exactly what
needs to be seen, and 30 fps gives 33 ms of granularity on an event that
decides the round. This is the strongest argument yet for the Pico: a
piezo disc on the target pad timestamps a hit in microseconds and cannot
be occluded. Scoring should be Pico's job, not the camera's.

**Timing intelligence replaces reaction speed.** A robot that strikes on a
fixed rhythm is trivially gamed — the human counts and leaves early. It
needs unpredictable commit timing, and ideally a read on when the human is
least ready (drift, settling, a glance away). That is low-rate vision, and
far easier than what the dodging robot needed. Feints become possible for
the first time, because the robot owns the clock.

## Safety

The robot is now deliberately swinging at a human hand. This is the
dominant design concern, not an afterthought.

1. **Cap strike distance in code**, ~8 cm. The one knob that improves
   speed and safety together.
2. **Lower `torque_limit` for strikes.** The current default of 800/1000
   is far too high for striking a person; the servo should yield on
   unexpected contact rather than push through it.
3. **Soft end effector** — a foam paddle, not the gripper. Compliance
   matters more than speed: the same 1 m/s impact is ~15 N spread over
   10 ms, or ~150 N over 1 ms against something rigid.
4. **Never command below the hand plane.** Position-limit the strike so
   the worst case is a gentle stall, not a press.
5. **Padded target pad** under the hand, so there is give on both sides.
   Striking a hand resting on a bare table is worse — no give underneath.
6. **Hardware e-stop** on the Pico, cutting torque independently of
   whether Python is responsive.

## Verified along the way

The joint-space straight line between the hover and contact IK solutions
stays within 6% of the Cartesian straight line (15.9 cm vs 15.0 cm for a
15 cm strike), so simple joint interpolation is safe for strikes of this
size and does not need Cartesian path planning. This was checked rather
than assumed, because a joint-space interpolation that wandered would be
a serious hazard for a striking robot.
