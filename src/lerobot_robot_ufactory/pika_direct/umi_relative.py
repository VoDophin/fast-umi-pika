"""Policy-independent UMI-style temporal representation."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Sequence

import numpy as np

from .geometry import relative_pose


@dataclass(frozen=True)
class UMIWindowConfig:
    observation_horizon: int = 2
    action_horizon: int = 16

    def __post_init__(self) -> None:
        if self.observation_horizon < 1 or self.action_horizon < 1:
            raise ValueError("observation_horizon and action_horizon must be positive")


class UMIRelativeEEProcessor:
    """Convert absolute TCP windows into current-EE-centred trajectories."""

    def __init__(self, config: UMIWindowConfig | None = None) -> None:
        self.config = config or UMIWindowConfig()

    def process_window(
        self,
        tcp_poses: Sequence[Sequence[float]],
        grippers: Sequence[float],
        current_index: int,
    ) -> dict[str, np.ndarray]:
        poses = np.asarray(tcp_poses, dtype=np.float64)
        grip = np.asarray(grippers, dtype=np.float64).reshape(-1)
        if poses.ndim != 2 or poses.shape[1] != 7 or len(grip) != len(poses):
            raise ValueError("tcp_poses must be Nx7 and grippers must contain N values")
        start = current_index - self.config.observation_horizon + 1
        stop = current_index + self.config.action_horizon
        if start < 0 or stop >= len(poses):
            raise IndexError("current_index does not have a complete past/future window")
        reference = poses[current_index]
        observation_indices = range(start, current_index + 1)
        action_indices = range(current_index + 1, stop + 1)
        observation_pose = np.stack([relative_pose(reference, poses[i]) for i in observation_indices])
        action_pose = np.stack([relative_pose(reference, poses[i]) for i in action_indices])
        observation_gripper = grip[start : current_index + 1, None]
        action_gripper = grip[current_index + 1 : stop + 1, None]
        return {
            "observation.relative_tcp": observation_pose.astype(np.float32),
            "observation.gripper_history": observation_gripper.astype(np.float32),
            "action": np.concatenate((action_pose, action_gripper), axis=-1).astype(np.float32),
        }

    def valid_current_indices(self, episode_length: int) -> range:
        return range(
            self.config.observation_horizon - 1,
            episode_length - self.config.action_horizon,
        )


class UMIDatasetMetadata:
    """Metadata proxy exposing policy-facing relative features and stats."""

    def __init__(self, raw_meta, features, stats, episodes) -> None:
        self._raw_meta = raw_meta
        self.features = features
        self.stats = stats
        self.episodes = episodes

    def __getattr__(self, name):
        return getattr(self._raw_meta, name)


class UMIRelativeDataset:
    """PyTorch-compatible view over an absolute-pose LeRobot dataset.

    The adapter indexes complete windows within episode boundaries and merges
    relative features into the current raw sample. It intentionally keeps the
    source dataset authoritative and unmodified.
    """

    def __init__(
        self,
        dataset,
        config: UMIWindowConfig | None = None,
        tcp_state_slice: slice = slice(7, 14),
        gripper_state_index: int = 14,
    ) -> None:
        self.dataset = dataset
        self.processor = UMIRelativeEEProcessor(config)
        self.tcp_state_slice = tcp_state_slice
        self.gripper_state_index = gripper_state_index
        self._windows: list[tuple[int, list[int]]] = []
        self._episode_ranges: list[tuple[int, int]] = []
        self._build_index()
        self.meta = self._build_meta()
        self.num_frames = len(self)
        self.num_episodes = len(self._episode_ranges)
        self.episodes = list(range(self.num_episodes))
        self.absolute_to_relative_idx = None

    @staticmethod
    def _scalar(value: Any) -> int:
        return int(value.item()) if hasattr(value, "item") else int(value)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _build_index(self) -> None:
        episodes: dict[int, list[int]] = {}
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            episodes.setdefault(self._scalar(sample["episode_index"]), []).append(index)
        for indices in episodes.values():
            indices.sort(key=lambda i: self._scalar(self.dataset[i]["frame_index"]))
            begin = len(self._windows)
            for current in self.processor.valid_current_indices(len(indices)):
                self._windows.append((current, indices))
            end = len(self._windows)
            if end > begin:
                self._episode_ranges.append((begin, end))

    def _build_meta(self):
        features = copy.deepcopy(self.dataset.meta.features)
        stats = copy.deepcopy(self.dataset.meta.stats)
        state_dim = self.processor.config.observation_horizon * 7
        features["observation.state"] = {
            "dtype": "float32", "shape": (state_dim,),
            "names": [f"umi_relative_state_{i}" for i in range(state_dim)],
        }
        features["action"] = {
            "dtype": "float32", "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        }
        episodes = {
            "dataset_from_index": [start for start, _ in self._episode_ranges],
            "dataset_to_index": [end for _, end in self._episode_ranges],
        }
        stats.update(self.compute_relative_stats())
        return UMIDatasetMetadata(self.dataset.meta, features, stats, episodes)

    def compute_relative_stats(self) -> dict[str, dict[str, Any]]:
        states, actions = [], []
        for current, indices in self._windows:
            raw_states = [self._numpy(self.dataset[i]["observation.state"]).reshape(-1) for i in indices]
            poses = [state[self.tcp_state_slice] for state in raw_states]
            grippers = [float(state[self.gripper_state_index]) for state in raw_states]
            result = self.processor.process_window(poses, grippers, current)
            states.append(np.concatenate((result["observation.relative_tcp"].reshape(-1), result["observation.gripper_history"].reshape(-1))))
            actions.append(result["action"])
        if not states:
            raise ValueError("dataset contains no complete UMI temporal window")

        def stats(values):
            values = np.asarray(values, dtype=np.float32)
            return {
                "mean": values.mean(axis=0), "std": values.std(axis=0),
                "min": values.min(axis=0), "max": values.max(axis=0),
                "q01": np.quantile(values, 0.01, axis=0).astype(np.float32),
                "q99": np.quantile(values, 0.99, axis=0).astype(np.float32),
            }
        return {
            "observation.state": stats(np.stack(states)),
            "action": stats(np.concatenate(actions, axis=0)),
        }

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        current, indices = self._windows[index]
        states = [self._numpy(self.dataset[i]["observation.state"]).reshape(-1) for i in indices]
        poses = [state[self.tcp_state_slice] for state in states]
        grippers = [float(state[self.gripper_state_index]) for state in states]
        relative = self.processor.process_window(poses, grippers, current)
        sample = dict(self.dataset[indices[current]])
        # Preserve images/task/index metadata while replacing the policy-facing
        # action with a current-frame-centred future trajectory.
        sample.update(relative)
        sample["observation.state"] = np.concatenate(
            (relative["observation.relative_tcp"].reshape(-1),
             relative["observation.gripper_history"].reshape(-1))
        ).astype(np.float32)
        return sample
