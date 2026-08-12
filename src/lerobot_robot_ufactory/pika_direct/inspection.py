from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .umi_relative import UMIRelativeEEProcessor, UMIWindowConfig


@dataclass
class InspectionReport:
    frame_count: int
    fps_estimate: float | None
    image_shapes: dict[str, tuple[int, ...]]
    tracker_position_range: tuple[list[float], list[float]]
    tcp_position_range: tuple[list[float], list[float]]
    gripper_range: tuple[float, float]
    timestamp_range: tuple[float, float]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _raw_components(item):
    state = _array(item["observation.state"]).reshape(-1)
    if len(state) < 18:
        raise ValueError("Pika direct observation.state must contain at least 18 named raw values")
    return state[0:7], state[7:14], state[14], state[16], state[17]


def inspect_samples(samples, max_translation_jump_m=0.25, max_rotation_jump_rad=1.5) -> InspectionReport:
    if not samples:
        raise ValueError("episode contains no frames")
    components = [_raw_components(item) for item in samples]
    tracker = np.stack([item[0] for item in components])
    tcp = np.stack([item[1] for item in components])
    gripper = np.asarray([item[2] for item in components])
    timestamps = np.asarray([item[3] for item in components])
    sample_ids = np.asarray([item[4] for item in components])
    frame_indices = np.asarray([int(_array(item["frame_index"]).reshape(-1)[0]) for item in samples])
    errors: list[str] = []
    warnings: list[str] = []
    if not all(np.all(np.isfinite(v)) for v in (tracker, tcp, gripper, timestamps)):
        errors.append("NaN or infinite sensor value")
    for name, poses in (("tracker", tracker), ("tcp", tcp)):
        norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-3):
            errors.append(f"invalid {name} quaternion norm")
    if len(np.unique(sample_ids)) != len(sample_ids):
        errors.append("duplicate sample_id")
    if np.any(np.diff(sample_ids) != 1):
        warnings.append("sensor sample_id gap (often caused by unsaved startup reads)")
    if len(np.unique(frame_indices)) != len(frame_indices):
        errors.append("duplicate frame_index")
    if np.any(np.diff(frame_indices) != 1):
        errors.append("missing frame_index")
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        errors.append("duplicate or non-monotonic timestamp")
    translation_jumps = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    if np.any(translation_jumps > max_translation_jump_m):
        warnings.append("large TCP translation jump")
    processor = UMIRelativeEEProcessor(UMIWindowConfig(1, 1))
    rotation_jumps = [
        np.linalg.norm(processor.process_window(tcp[i - 1 : i + 1], gripper[i - 1 : i + 1], 0)["action"][0, 3:6])
        for i in range(1, len(tcp))
    ]
    if any(value > max_rotation_jump_rad for value in rotation_jumps):
        warnings.append("large TCP rotation jump")
    image_shapes = {
        key.removeprefix("observation.images."): tuple(_array(value).shape)
        for key, value in samples[0].items()
        if key.startswith("observation.images.")
    }
    fps = None if len(dt) == 0 or np.median(dt) <= 0 else float(1.0 / np.median(dt))
    return InspectionReport(
        len(samples), fps, image_shapes,
        (tracker[:, :3].min(0).tolist(), tracker[:, :3].max(0).tolist()),
        (tcp[:, :3].min(0).tolist(), tcp[:, :3].max(0).tolist()),
        (float(gripper.min()), float(gripper.max())),
        (float(timestamps.min()), float(timestamps.max())), errors, warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Pika direct LeRobot episode")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--timestep", type=int)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    samples = [dataset[i] for i in range(len(dataset)) if int(dataset[i]["episode_index"]) == args.episode]
    report = inspect_samples(samples)
    print(report)
    if args.timestep is not None:
        components = [_raw_components(item) for item in samples]
        poses = [item[1] for item in components]
        grippers = [item[2] for item in components]
        processor = UMIRelativeEEProcessor(UMIWindowConfig(args.observation_horizon, args.action_horizon))
        print("absolute_tcp:", poses[args.timestep])
        print("relative_window:", processor.process_window(poses, grippers, args.timestep))
