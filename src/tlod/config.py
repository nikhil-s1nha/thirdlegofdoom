"""Configuration.

One YAML file decides whether the same code drives a simulator or a real
arm, which camera it looks through, and how hard it is allowed to move.
Anything a person might reasonably want to change between runs lives here
rather than in a constructor argument buried three layers down.

Defaults are the simulator, deliberately. Running the wrong thing should
mean nothing moves, not that an arm swings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ArmConfig:
    backend: str = "mock"                 # "mock" | "feetech"
    port: str = ""                        # serial port; empty = autodetect
    baudrate: int = 1_000_000
    calibration: str = ""                 # path; empty = identity mapping
    lerobot_id: str = ""                  # read lerobot-calibrate output instead
    goal_acceleration: int = 60
    torque_limit: int = 800
    # Simulator dynamics. Estimates from the STS3215 datasheet until M6
    # measures them; see docs/ROADMAP.md.
    sim_max_speed: float = 3.5
    sim_accel: float = 25.0
    sim_latency: float = 0.004


@dataclass(slots=True)
class SafetyConfig:
    max_speed: float = 2.0
    strike_speed: float = 5.0
    joint_margin: float = 0.05
    table_z: float = 0.0
    min_height: float = 0.015
    max_radius: float = 0.33
    min_radius: float = 0.08
    max_height: float = 0.45
    command_timeout: float = 0.5


@dataclass(slots=True)
class CameraConfig:
    source: str = "mock"                  # "mock" | "opencv"
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    fourcc: str = "MJPG"
    latency_offset: float | None = None   # None = estimate from frame period
    autofocus: bool = False
    autoexposure: bool = False
    exposure: float | None = None
    intrinsics: str = ""                  # .npz path; empty = approximate
    extrinsics: str = ""                  # .npz path; empty = synthetic pose
    hfov_deg: float = 70.0
    # Where the camera sits, when no calibration file exists yet.
    position: tuple[float, float, float] = (0.15, -0.45, 0.55)
    look_at: tuple[float, float, float] = (0.22, 0.0, 0.0)


@dataclass(slots=True)
class VisionConfig:
    detector: str = "mediapipe"           # "mediapipe" | "scripted" | "none"
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    delegate: str = "cpu"
    depth_mode: str = "auto"              # "auto" | "size" | "plane"
    hand_height: float = 0.06
    palm_width_m: float = 0.081
    detect_objects: bool = False
    process_noise: float = 4.0
    measurement_noise: float = 0.012


@dataclass(slots=True)
class RuntimeConfig:
    control_hz: float = 100.0
    perception_max_age: float = 0.25
    # Set from a measurement (`tlod bench loop`), not from taste.
    prediction_horizon: float = 0.30
    policy: str = "idle"


@dataclass(slots=True)
class Config:
    arm: ArmConfig = field(default_factory=ArmConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        sections = {
            "arm": ArmConfig,
            "safety": SafetyConfig,
            "camera": CameraConfig,
            "vision": VisionConfig,
            "runtime": RuntimeConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            fields = {f for f in klass.__dataclass_fields__}
            given = data.get(name) or {}
            unknown = set(given) - fields
            if unknown:
                # Loud, because a typo in a config key that is silently
                # ignored is how a safety limit fails to apply.
                raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
            kwargs[name] = klass(**given)
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        data = asdict(self)
        for name, values in sections.items():
            data[name].update(values)
        return Config.from_dict(data)
