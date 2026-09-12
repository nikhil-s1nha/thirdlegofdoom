# Running vision on a board with no screen

You cannot check whether the boxes land on the hand by looking, so check
it with numbers. `tlod vision-check` does that and exits non-zero on
failure, so it works from cron as well as from a terminal.

## Precision is not accuracy

The distinction matters more than any single number here.

**Precision** — is the estimate stable and self-consistent? Measurable
with nothing but a camera.

**Accuracy** — is the estimate correct? Needs ground truth. A camera
alone cannot tell you: a badly calibrated camera produces beautifully
precise, consistently wrong positions, and every camera-only check passes
while the arm reaches 5 cm to the left of everything.

The arm is the only ground truth on the robot. Forward kinematics knows
where the gripper is to within a millimetre, so driving it to known
configurations and comparing what vision reports is the one check that
catches a bad extrinsic.

## Two commands

```bash
# precision — camera only, no motion
tlod vision-check --duration 30

# accuracy — drives the arm, needs a green marker on the gripper
tlod vision-check --with-arm --duration 20 --json report.json
```

Put a hand in view and move it slowly around the frame for the precision
run. For the accuracy run, clear the workspace.

## The checks

| check | default limit | what a failure means |
|---|---|---|
| detection rate | 60% of frames | lighting, hand out of frame, or the model is not loading |
| frame-to-frame jitter | 25 mm | noisy detection, motion blur, or exposure hunting |
| teleports | 5% of frames | detector losing and reacquiring — usually occlusion or frame edge |
| depth spread | 80 mm | wrong focal length in the intrinsics |
| accuracy vs kinematics | 30 mm | bad extrinsics, or a wrong `palm_width_m` |

Jitter is measured from consecutive-frame differences, not spread about a
mean, so a slow legitimate drift does not read as noise. Hold your hand
still for a meaningful number.

What this rig actually gets, for calibration against your own: 31 fps,
100% detection with a hand in frame, 2.4–5 mm jitter, and 12.1 mm on
`--with-arm`. That last number is larger than the other accuracy checks
in this project (`scripts/check_extrinsics.py` says 4.5 mm, `tlod touch
--real` puts the tool 2.5 mm from a detected object) and the reason is
the marker: `--with-arm` sticks it on the gripper *jaw*, while forward
kinematics reports the tool point. It is scoring a real offset between
two different places on the same gripper, not extra error. Do not tune
the extrinsics to make this number small.

Depth spread is **reported but not enforced** unless you pass
`--fixed-distance`. Its precondition — that you held the hand at a
constant distance — is an instruction to you, not something the code can
verify, and enforcing it by default fails honest runs where the hand
really did move in depth.

Thresholds live in `Thresholds` in `tlod/vision/check.py` and are
deliberately loose. Tighten them once you know what your setup achieves.

## Reading the output

```
  frames 1804, hands detected in 1731 (96%), 29.6 fps

  PASS  detection rate                96.00 %    (limit 60)
  PASS  frame-to-frame jitter          4.20 mm   (limit 25)
  PASS  teleports                      0.30 %    (limit 5)
  PASS  depth spread                  41.00 mm   (limit 80)
  FAIL  accuracy vs kinematics        52.10 mm   (limit 30)  - mean 48.3 mm over 8 poses
```

That pattern — every camera check passing, accuracy failing — is the
signature of a calibration problem, not a vision problem. The detector is
working fine and reporting the wrong place. Recalibrate extrinsics
(`tlod calibrate extrinsics`) and check the marker was the only green
thing in frame.

The reverse — low detection rate but good accuracy on what it does find —
is a lighting or exposure problem.

`--json` writes the same thing machine-readably, including per-pose
`truth` / `seen` / `error_mm` for the arm points. A consistent error in
one direction across all poses is an offset; errors growing with distance
from the base suggest a scale problem in the intrinsics.

## Seeing what it sees

When the numbers are wrong and you want to look:

```bash
tlod vision-serve --preview 8081 --to <control board>
```

Open `http://<vision board>:8081/` from any machine. Plain MJPEG, so
every browser renders it with no player or plugin, and `curl` can save
it. Encoding costs a few ms per frame, so it is throttled and runs on its
own thread — but leave it off in normal operation.

`--save-frames DIR` on `vision-check` writes annotated JPEGs instead:
palm ring, landmark dots, and the computed 3D position drawn on each.
Useful over `scp` when you cannot reach a port.

Both calibration steps take `--preview PORT` for the same reason:

```bash
tlod calibrate intrinsics --fisheye --preview 8080
tlod calibrate extrinsics --marker red --preview 8080
```

## Four smaller views

Bringing the camera up on a headless board produced four diagnostics that
each answer one question the main commands answer badly or not at all.
All of them serve an annotated MJPEG stream on port 8080 — open it from a
laptop on the same network. Each has a docstring explaining the failure
it exists for; that is the fuller version of this table.

| | question |
|---|---|
| `scripts/board_view.py [pattern] [index]` | is the chessboard being detected *at all*? |
| `scripts/marker_view.py <colour> [index]` | which blob would the calibrator pick? |
| `scripts/calib_view.py <pattern> [square] [index]` | is the lens model describing this lens? |
| `scripts/check_extrinsics.py <colour> [config]` | does the camera agree with the arm, in mm? |

**`board_view.py`** exists because `tlod calibrate intrinsics`
auto-captures, so it prints nothing at all when detection never fires —
which is indistinguishable from a dead camera. Pass `scan` instead of a
pattern and it tries every plausible board size and tells you which one
matches; counting inner corners off a printout is easy to get wrong
twice.

**`marker_view.py`** exists because the calibrator takes the *largest*
blob of the marker colour and believes it is the gripper. It cannot tell
a marker from a mug, and it does not fail when it picks wrong — it
produces a confident calibration of the camera against the mug, after the
arm has already driven a dozen poses. Look first, calibrate second.

**`calib_view.py`** shows the raw frame beside the undistorted one, so
you can see whether straight lines in the room are straight in the
corners of the corrected image. An RMS number says a calibration is bad
without saying why; this separates a sloppy shoot from a lens the model
cannot describe. A wide lens usually needs `--fisheye` rather than more
views — the fisheye fit here came out at 0.165 px RMS where the pinhole
model could not fit at all.

**`check_extrinsics.py`** drives the arm to poses the calibration never
saw, projects where forward kinematics says the tool is, finds the
marker, and reports the gap in millimetres as well as pixels. An
extrinsics RMS is in pixels and is computed from the points that produced
the fit, so it says how well the solve explained its own data. This asks
the useful question instead. It moves the arm; clear the workspace.

## One camera cannot measure depth

The largest error source in this system is not the lens or the
calibration — both are now good to a few millimetres — it is how far away
the hand is. A single camera has no way to measure that, and
`depth_mode: size` infers it from apparent palm width, which is a
proportional error on a quantity that varies between people.

`depth_mode: plane` with the palm flat on the table removes it entirely:
the height is then known rather than estimated, and the pixel ray meets a
known plane. If your game can ask for a flat hand, ask for it.

## The `/dev/video*` index is not stable

It changes across reboots and replugs — twice in one evening here, with
`/dev/video5` a hardware encoder on one boot and the camera on the next.
Nothing warns you; a wrong index reads as a vision problem. `tlod
cameras` lists what opens, `scripts/board_view.py` shows what each one is
looking at, and `/dev/v4l/by-id/` gives a name that survives a reboot.

## From cron

```cron
*/30 * * * * cd /home/tlod/thirdlegofdoom && .venv/bin/tlod vision-check \
    --duration 20 --json /var/log/tlod-vision.json || logger -t tlod "vision degraded"
```

Leave `--with-arm` out of anything unattended: it moves the arm.

Watching the JSON over time catches drift that a single run cannot —
detection rate falling as a lens fogs, jitter climbing as the light goes,
accuracy creeping as a mount loosens.
