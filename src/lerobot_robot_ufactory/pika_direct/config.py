from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@dataclass
class TrackerToTCPConfig:
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@RobotConfig.register_subclass("uf::pika_direct")
@dataclass
class PikaDirectRobotConfig(RobotConfig):
    port: str | None = None
    tracker_device_id: str | None = None
    tracker_to_tcp: TrackerToTCPConfig = field(default_factory=TrackerToTCPConfig)
    gripper_closed_width_mm: float = 0.0
    gripper_open_width_mm: float = 100.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gripper_open_width_mm <= self.gripper_closed_width_mm:
            raise ValueError("gripper_open_width_mm must exceed gripper_closed_width_mm")
