"""Policy-independent UMI-style temporal representation."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from collections import deque
from typing import Any, Sequence

import numpy as np

from .geometry import relative_pose, relative_rotvec_pose_to_absolute


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


class UMIRelativeDataset:   # 训练时读取时临时转换格式
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
    # 最终的数据格式：observation: ((6+1)*2,); action: (16,6+1); 其中6是相对TCP，1是夹爪开合度
    # 6 表示旋转向量；


class UMIDiffusionRelativeDataset:
    """Explicit-time FastUMI view over a LeRobot Diffusion dataset.

    The wrapped LeRobot dataset must already have applied Diffusion delta
    indices.  Its observation window has ``Hobs`` steps and its temporary
    action window has ``Hobs + Hact`` steps, beginning at ``t-Hobs+1``.
    This adapter re-expresses poses around ``T_t`` and keeps only the strictly
    future action interval ``[t+1, ..., t+Hact]``.
    """

    def __init__(
        self,
        dataset,
        config: UMIWindowConfig | None = None,
        tcp_state_slice: slice = slice(7, 14),
        gripper_state_index: int = 14,
        action_pose_slice: slice = slice(0, 7),
        action_gripper_index: int = 7,
    ) -> None:
        self.dataset = dataset
        self.config = config or UMIWindowConfig()
        self.tcp_state_slice = tcp_state_slice
        self.gripper_state_index = gripper_state_index
        self.action_pose_slice = action_pose_slice
        self.action_gripper_index = action_gripper_index
        self.meta = self._build_meta()
        self.num_frames = len(dataset)
        num_episodes = getattr(dataset, "num_episodes", None)
        if num_episodes is None:
            selected_episodes = getattr(dataset, "episodes", None)
            num_episodes = len(selected_episodes) if selected_episodes is not None else len(
                self.meta.episodes["dataset_from_index"]
            )
        self.num_episodes = num_episodes
        self.episodes = getattr(dataset, "episodes", None)
        self.absolute_to_relative_idx = getattr(dataset, "absolute_to_relative_idx", None)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _stats(values: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("relative statistics require a non-empty NxD array")
        return {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "q01": np.quantile(values, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(values, 0.99, axis=0).astype(np.float32),
        }

    def _transform_arrays(
        self, state_window: Any, action_window: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        states = self._numpy(state_window).astype(np.float64, copy=False)
        actions = self._numpy(action_window).astype(np.float64, copy=False)
        hobs = self.config.observation_horizon
        hact = self.config.action_horizon
        if states.ndim != 2 or states.shape[0] != hobs:
            raise ValueError(f"Diffusion observation.state must have shape ({hobs}, state_dim)")
        if actions.ndim != 2 or actions.shape[0] < hobs + hact:
            raise ValueError(
                f"Diffusion source action window must contain at least {hobs + hact} steps"
            )
        reference = states[-1, self.tcp_state_slice]
        relative_states = np.stack(
            [relative_pose(reference, pose) for pose in states[:, self.tcp_state_slice]]
        )
        state_grippers = states[:, self.gripper_state_index, None]
        future = actions[hobs : hobs + hact]
        relative_actions = np.stack(
            [relative_pose(reference, pose) for pose in future[:, self.action_pose_slice]]
        )
        action_grippers = future[:, self.action_gripper_index, None]
        return (
            np.concatenate((relative_states, state_grippers), axis=-1).astype(np.float32),
            np.concatenate((relative_actions, action_grippers), axis=-1).astype(np.float32),
        )

    def _numeric_rows(self):
        """Read parquet/HF rows directly so statistics never decode videos."""
        source = getattr(self.dataset, "hf_dataset", None)
        if source is None and hasattr(self.dataset, "_ensure_hf_dataset_loaded"):
            self.dataset._ensure_hf_dataset_loaded()
            source = getattr(self.dataset, "hf_dataset", None)
        if source is not None:
            for index in range(len(source)):
                yield source[index]
            return
        for index in range(len(self.dataset)):
            yield self.dataset[index]

    def compute_relative_stats(self) -> dict[str, dict[str, np.ndarray]]:
        episodes: dict[int, list[dict[str, Any]]] = {}
        for row in self._numeric_rows():
            episode_value = self._numpy(row["episode_index"]).reshape(-1)[0]
            episodes.setdefault(int(episode_value), []).append(row)

        relative_states: list[np.ndarray] = []
        relative_actions: list[np.ndarray] = []
        hobs = self.config.observation_horizon
        hact = self.config.action_horizon
        for rows in episodes.values():
            rows.sort(key=lambda row: int(self._numpy(row["frame_index"]).reshape(-1)[0]))
            poses = [self._numpy(row["observation.state"])[self.tcp_state_slice] for row in rows]
            state_grippers = [
                float(self._numpy(row["observation.state"])[self.gripper_state_index]) for row in rows
            ]
            absolute_actions = [self._numpy(row["action"]) for row in rows]
            for current, reference in enumerate(poses):
                for history in range(max(0, current - hobs + 1), current + 1):
                    relative_states.append(
                        np.concatenate(
                            (relative_pose(reference, poses[history]), [state_grippers[history]])
                        ).astype(np.float32)
                    )
                for future in range(current + 1, min(len(rows), current + hact + 1)):
                    action = absolute_actions[future]
                    relative_actions.append(
                        np.concatenate(
                            (
                                relative_pose(reference, action[self.action_pose_slice]),
                                [float(action[self.action_gripper_index])],
                            )
                        ).astype(np.float32)
                    )
        if not relative_actions:
            raise ValueError("dataset contains no non-padded future Diffusion actions")
        return {
            "observation.state": self._stats(np.stack(relative_states)),
            "action": self._stats(np.stack(relative_actions)),
        }

    def _build_meta(self):
        features = copy.deepcopy(self.dataset.meta.features)
        stats = copy.deepcopy(self.dataset.meta.stats)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        }
        stats.update(self.compute_relative_stats())
        return UMIDatasetMetadata(self.dataset.meta, features, stats, self.dataset.meta.episodes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        state, action = self._transform_arrays(sample["observation.state"], sample["action"])
        hobs = self.config.observation_horizon
        hact = self.config.action_horizon
        source_action_pad = self._numpy(sample.get("action_is_pad", np.zeros(hobs + hact, dtype=bool)))
        if source_action_pad.shape[0] < hobs + hact:
            raise ValueError("source action_is_pad is shorter than the sampled action window")
        sample["observation.state"] = state
        sample["action"] = action
        sample["action_is_pad"] = source_action_pad[hobs : hobs + hact].astype(bool)
        return sample


class UMIDiffusionInferenceAdapter:
    """Build current-frame-centred Diffusion windows during online inference."""

    TCP_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")

    def __init__(
        self,
        policy,
        config: UMIWindowConfig | None = None,
        preprocessor=None,
        postprocessor=None,
        camera_keys: Sequence[str] | None = None,
        tcp_state_slice: slice = slice(7, 14),
        gripper_state_index: int = 14,
    ) -> None:
        self.policy = policy
        self.config = config or UMIWindowConfig()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        if camera_keys is None:
            policy_config = getattr(policy, "config", None)
            image_features = getattr(policy_config, "image_features", {})
            camera_keys = tuple(image_features) or None
        self.camera_keys = tuple(camera_keys) if camera_keys is not None else None
        self.tcp_state_slice = tcp_state_slice
        self.gripper_state_index = gripper_state_index
        self._history: deque[tuple[np.ndarray, float, dict[str, Any]]] = deque(
            maxlen=self.config.observation_horizon
        )

    def reset(self) -> None:
        self._history.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if processor is not None and hasattr(processor, "reset"):
                processor.reset()

    def _extract(self, observation: dict[str, Any]) -> tuple[np.ndarray, float, dict[str, Any]]:
        if "observation.state" in observation:
            state = self._numpy(observation["observation.state"]).reshape(-1)
            tcp = state[self.tcp_state_slice].astype(np.float64)
            gripper = float(state[self.gripper_state_index])
        else:
            tcp = np.asarray([observation[f"tcp.{name}"] for name in self.TCP_NAMES], dtype=np.float64)
            gripper = float(observation["gripper"])
        camera_keys = self.camera_keys or tuple(
            key for key in observation if key.startswith("observation.images.")
        )
        images = {}
        for key in camera_keys:
            source_key = key
            if source_key not in observation and key.startswith("observation.images."):
                source_key = key.removeprefix("observation.images.")
            if source_key not in observation:
                raise KeyError(f"camera {key!r} is missing from the online observation")
            images[key] = self._prepare_image(observation[source_key])
        return tcp, gripper, images

    @staticmethod
    def _prepare_image(image: Any) -> Any:
        """Normalize raw HWC uint8 or preserve model-ready CHW float images."""
        import torch

        tensor = image.detach().clone() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        if tensor.ndim != 3:
            raise ValueError("camera image must have three dimensions")
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        elif tensor.shape[0] not in (1, 3, 4):
            raise ValueError("camera image must be HWC or CHW")
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32).div_(255.0)
        else:
            tensor = tensor.to(torch.float32)
        return tensor

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def append_observation(self, observation: dict[str, Any]) -> None:
        frame = self._extract(observation)
        if not self._history:
            for _ in range(self.config.observation_horizon - 1):
                self._history.append(frame)
        self._history.append(frame)

    def _build_window(self, add_batch_dimension: bool) -> dict[str, Any]:
        if not self._history:
            raise RuntimeError("append_observation() must be called before build_batch()")
        import torch

        frames = list(self._history)
        reference = frames[-1][0]
        states = np.stack(
            [np.concatenate((relative_pose(reference, tcp), [gripper])) for tcp, gripper, _ in frames]
        ).astype(np.float32)
        batch: dict[str, Any] = {"observation.state": torch.from_numpy(states)}
        for key in frames[-1][2]:
            images = [
                image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
                for _, _, frame_images in frames
                for image in [frame_images[key]]
            ]
            batch[key] = torch.stack(images, dim=0)
        if add_batch_dimension:
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}
        return batch

    def build_batch(self) -> dict[str, Any]:
        """Return the policy-ready explicit window with a leading batch axis."""
        return self._build_window(add_batch_dimension=True)

    def predict_relative_action(self, observation: dict[str, Any]) -> Any:
        """Return the unnormalized first strictly-future relative action."""
        import torch

        self.append_observation(observation)
        # LeRobot's official Diffusion preprocessor adds the batch dimension.
        batch = self._build_window(add_batch_dimension=self.preprocessor is None)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        else:
            try:
                device = next(self.policy.parameters()).device
                batch = {key: value.to(device) for key, value in batch.items()}
            except (AttributeError, StopIteration):
                pass
        image_feature_keys = tuple(getattr(self.policy.config, "image_features", {}))
        if image_feature_keys:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in image_feature_keys], dim=-4
            )
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        with torch.no_grad():
            actions = self.policy.diffusion.generate_actions(batch)
        first_action = actions[:, 0]
        if self.postprocessor is not None:
            first_action = self.postprocessor(first_action)
        return first_action

    @staticmethod
    def decode_action(reference_tcp: Sequence[float], relative_action: Any) -> dict[str, Any]:
        action = UMIDiffusionRelativeDataset._numpy(relative_action).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("relative action must contain seven finite values")
        return {
            "tcp_pose": relative_rotvec_pose_to_absolute(reference_tcp, action[:6]),
            "gripper": float(np.clip(action[6], 0.0, 1.0)),
        }

    def predict_absolute_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        relative = self.predict_relative_action(observation)
        reference_tcp = self._history[-1][0]
        return self.decode_action(reference_tcp, relative)
