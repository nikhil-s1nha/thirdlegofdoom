# Why the robot slaps instead of dodging

## The latency problem

Hand moves to arm moves, end to end:

| stage | cost |
|---|---|
| camera → decode | 30–50 ms |
| MediaPipe hand landmarks | 19 ms measured |
| IK + control | 0.4 ms measured |
| serial write @1 Mbaud | 0.5 ms |
| servo response and travel | 120–300 ms |
| **total** | **200–370 ms** |

Human visual reaction is 200–250 ms *plus* hand travel. A robot reacting
to what it saw is always late, and the servo alone eats most of the
budget, so no optimisation fixes it.

## The fix: initiate, don't respond

Latency taxes whoever is responding. If the robot strikes and the human
dodges, the robot's pipeline delay is spent *before* the strike, where
nobody is waiting on it. A hand waiting to be slapped is also nearly
stationary, so a slightly stale position estimate is still correct.

This is why prediction is not load-bearing here. The Kalman filter stays
for smoothing, for a settled/moving signal, and for uncertainty gating —
the game extrapolates once, by about 100 ms, as a small lead correction.

## Then dodging turned out to be a cliff

Measured against a simulated 250 ms opponent, feints off:

| hover | strike | robot wins |
|---|---|---|
| 6 cm | 180 ms | 100% |
| 8 cm | 250 ms | 100% |
| 8 cm | 350 ms | 100% |
| 10 cm | 450 ms | 17% |
| 12 cm | 550 ms | 0% |

Always-wins to never-wins inside 100 ms. That is a step function, not a
difficulty curve, and no parameter turns it into a game.

The window is much smaller than the strike duration because contact fires
at ~70% of the travel (the paddle reaches the hand before the motion
ends) and motion onset costs the first ~25% (min-jerk spends a quarter of
its time covering the first few millimetres, and the servo ramps on top).
So the human gets ~45% of the strike duration. Beating an 8 cm strike
needs a sub-70 ms reaction.

Slowing the arm to compensate takes ~650 ms per strike, which stops
reading as a slap. Striking from further away is both slower *and*
harder-hitting.

## What makes it a game: feints

Real hand-slap is slapper-favoured too. What makes it fun is that the
dodger is punished for flinching.

| | point to |
|---|---|
| strike lands | robot |
| strike dodged | human |
| feint draws a flinch | robot |
| feint held through | human |

The human reads intent rather than racing physics. Difficulty becomes
feint frequency — a smooth dial — and the arm keeps striking short, fast
and softly, so safety stops fighting the game design.

Calibrated (`tlod eval`), robot win rate:

| preset | vs 180 ms | vs 250 ms | vs 350 ms |
|---|---|---|---|
| easy | 21% | 0% | 0% |
| normal | 57% | 43% | 50% |
| hard | 86% | 79% | 64% |

Re-tune against real people; these are against the simulated opponent.

## Impact

An 8 cm strike lands in ~210 ms at ~0.7 m/s. A casual human high-five is
1–3 m/s. Short strikes are better on both axes at once — faster to land
and softer on impact — which gives the design rule: hover close, strike
short.

Safety specifics are in the walkthrough. The short version: cap
`max_drop`, lower `torque_limit` during strikes, foam paddle not the
gripper, and a hardware e-stop.
