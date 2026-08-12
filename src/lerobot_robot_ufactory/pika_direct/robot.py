from __future__ import annotations

import time
from typing import Any

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot_robot_ufactory.devices.pika import PikaDevice

from .config import PikaDirectRobotConfig
from .geometry import compose_pose, normalize_quaternion


class InvalidPikaFrame(RuntimeError):
    """A transient sensor frame that must be dropped, never fabricated."""


class PikaDirectRobot(Robot):
    """Read-only LeRobot adapter for handheld Pika demonstrations."""

    config_class = PikaDirectRobotConfig
    name = "pika_direct"

    def __init__(self, config: PikaDirectRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._device: PikaDevice | None = None
        self._sense = None
        self._is_connected = False
        self._sample_id = 0
        self._last_action: dict[str, float] | None = None
        self.invalid_frame_count = 0
        extrinsic = config.tracker_to_tcp
        self._tracker_to_tcp = np.asarray(
            (*extrinsic.translation_m, *extrinsic.rotation_quaternion_xyzw), dtype=np.float64
        )
        # Validate calibration immediately.
        normalize_quaternion(self._tracker_to_tcp[3:])

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        pose_names = ("x", "y", "z", "qx", "qy", "qz", "qw")
        # LeRobot 0.4.3 packs scalar robot features, in insertion order, into
        # the standard observation.state tensor and stores these names in the
        # dataset metadata. This preserves every raw component while remaining
        # directly consumable by existing policies.
        features: dict[str, type | tuple[int, ...]] = {
            **{f"tracker.{name}": float for name in pose_names},
            **{f"tcp.{name}": float for name in pose_names},
            "gripper": float,
            "gripper_width_mm": float,
            "sensor_timestamp": float,
            "sample_id": float,
        }
        for key, camera in self.cameras.items():
            features[key] = (camera.height, camera.width, 3)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"tcp.{name}": float for name in ("x", "y", "z", "qx", "qy", "qz", "qw")},
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return
        self._device = PikaDevice(
            1,
            pika_sense_port=self.config.port,
            pika_tracker_device=self.config.tracker_device_id,
        )
        self._sense = self._device.pika_sense
        for camera in self.cameras.values():
            camera.connect()
        self._is_connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _normalize_gripper(self, width_mm: float) -> float:
        span = self.config.gripper_open_width_mm - self.config.gripper_closed_width_mm
        return float(np.clip((width_mm - self.config.gripper_closed_width_mm) / span, 0.0, 1.0))

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected or self._sense is None:
            raise DeviceNotConnectedError("PikaDirectRobot must be connected before reading")
        pose = self._sense.get_pose(self._device.pika_tracker_device)
        if pose is None:
            self.invalid_frame_count += 1
            raise InvalidPikaFrame("Pika tracker returned no pose")
        tracker_pose = np.asarray((*pose.position, *pose.rotation), dtype=np.float64)
        if tracker_pose.shape != (7,) or not np.all(np.isfinite(tracker_pose)):
            self.invalid_frame_count += 1
            raise InvalidPikaFrame("Pika tracker returned an invalid pose")
        tracker_pose[3:] = normalize_quaternion(tracker_pose[3:])
        tcp_pose = compose_pose(tracker_pose, self._tracker_to_tcp)
        width = self._sense.get_gripper_distance()
        if width is None or not np.isfinite(float(width)):
            self.invalid_frame_count += 1
            raise InvalidPikaFrame("Pika gripper returned an invalid width")
        width = float(width)
        gripper = self._normalize_gripper(width)
        timestamp = time.monotonic()
        self._sample_id += 1
        observation: dict[str, Any] = {
            **{f"tracker.{name}": float(value) for name, value in zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"), tracker_pose, strict=True
            )},
            **{f"tcp.{name}": float(value) for name, value in zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"), tcp_pose, strict=True
            )},
            "gripper": gripper,
            "gripper_width_mm": width,
            "sensor_timestamp": timestamp,
            "sample_id": float(self._sample_id),
        }
        for key, camera in self.cameras.items():
            image = camera.async_read()
            if image is None:
                self.invalid_frame_count += 1
                raise InvalidPikaFrame(f"camera {key!r} returned no frame")
            observation[key] = image
        names = ("x", "y", "z", "qx", "qy", "qz", "qw")
        self._last_action = {f"tcp.{name}": float(value) for name, value in zip(names, tcp_pose, strict=True)}
        self._last_action["gripper"] = gripper
        return observation

    def action_from_observation(self, observation: dict[str, Any]) -> dict[str, float]:
        """Return the cached action paired with the most recent atomic sample."""
        if self._last_action is None:
            raise RuntimeError("get_observation() must be called before action_from_observation()")
        return self._last_action.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Compatibility no-op: this acquisition device never controls hardware."""
        return action

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            camera.disconnect()
        if self._sense is not None:
            self._sense.disconnect()
        self._sense = None
        self._device = None
        self._is_connected = False
