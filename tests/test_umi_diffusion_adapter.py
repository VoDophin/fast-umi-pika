import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src/lerobot_robot_ufactory/pika_direct"
package = types.ModuleType("pika_diffusion_adapter_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for module_name in ("geometry", "umi_relative"):
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.{module_name}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
geometry = sys.modules["pika_diffusion_adapter_test.geometry"]
umi = sys.modules["pika_diffusion_adapter_test.umi_relative"]


def raw_row(frame, episode=0):
    tracker = [0, 0, 0, 0, 0, 0, 1]
    tcp = [float(frame), 0, 0, 0, 0, 0, 1]
    state = np.asarray([*tracker, *tcp, frame / 10, 50, frame / 30, frame], dtype=np.float32)
    return {
        "observation.state": state,
        "action": np.asarray([*tcp, frame / 10], dtype=np.float32),
        "episode_index": np.asarray(episode),
        "frame_index": np.asarray(frame),
    }


class FakeDeltaDataset:
    """Mimic LeRobot output for Hobs=2 and temporary horizon=4."""

    def __init__(self):
        class Meta:
            features = {
                "observation.state": {"dtype": "float32", "shape": (18,)},
                "observation.images.pika": {
                    "dtype": "video", "shape": (3, 4, 4),
                    "names": ["channels", "height", "width"],
                },
                "action": {"dtype": "float32", "shape": (8,)},
            }
            stats = {
                "observation.images.pika": {
                    key: np.zeros((3, 1, 1), dtype=np.float32)
                    for key in ("mean", "std", "min", "max", "q01", "q99")
                }
            }
            camera_keys = ["observation.images.pika"]
            episodes = {"dataset_from_index": [0], "dataset_to_index": [5]}

        self.meta = Meta()
        self.hf_dataset = [raw_row(frame) for frame in range(5)]
        self.episodes = None
        self.num_episodes = 1
        self.absolute_to_relative_idx = None

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        obs_indices = [max(0, index - 1), index]
        # Temporary Diffusion deltas are [-1, 0, 1, 2].
        action_indices = [max(0, min(4, index + delta)) for delta in (-1, 0, 1, 2)]
        return {
            "observation.state": np.stack(
                [self.hf_dataset[i]["observation.state"] for i in obs_indices]
            ),
            "observation.images.pika": np.stack(
                [np.full((3, 4, 4), i, dtype=np.float32) for i in obs_indices]
            ),
            "observation.state_is_pad": np.asarray([index == 0, False]),
            "observation.images.pika_is_pad": np.asarray([index == 0, False]),
            "action": np.stack([self.hf_dataset[i]["action"] for i in action_indices]),
            "action_is_pad": np.asarray(
                [index + delta < 0 or index + delta >= 5 for delta in (-1, 0, 1, 2)]
            ),
            "episode_index": np.asarray(0),
            "frame_index": np.asarray(index),
        }


def test_diffusion_adapter_keeps_explicit_time_and_strictly_future_actions():
    wrapped = umi.UMIDiffusionRelativeDataset(
        FakeDeltaDataset(), umi.UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )
    item = wrapped[2]
    assert item["observation.state"].shape == (2, 7)
    assert item["observation.images.pika"].shape == (2, 3, 4, 4)
    assert item["action"].shape == (2, 7)
    assert wrapped.meta.features["observation.state"]["shape"] == (7,)
    assert wrapped.meta.features["action"]["shape"] == (7,)
    np.testing.assert_allclose(item["observation.state"][:, 0], [-1, 0])
    np.testing.assert_allclose(item["observation.state"][-1, :6], np.zeros(6), atol=1e-7)
    np.testing.assert_allclose(item["action"][:, 0], [1, 2])
    assert not item["action_is_pad"].any()


def test_diffusion_adapter_preserves_native_boundary_padding_masks():
    wrapped = umi.UMIDiffusionRelativeDataset(
        FakeDeltaDataset(), umi.UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )
    first = wrapped[0]
    last = wrapped[4]
    assert first["observation.state_is_pad"].tolist() == [True, False]
    assert first["observation.images.pika_is_pad"].tolist() == [True, False]
    assert first["action_is_pad"].tolist() == [False, False]
    assert last["action_is_pad"].tolist() == [True, True]
    # Statistics exclude copied padding actions: the largest genuine relative
    # future displacement in this episode is two frames.
    assert wrapped.meta.stats["action"]["max"][0] == 2


def test_online_adapter_rebases_absolute_history_on_every_frame():
    class Policy:
        def reset(self):
            self.was_reset = True

    adapter = umi.UMIDiffusionInferenceAdapter(
        Policy(), umi.UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )

    def observation(x):
        state = np.asarray([*[0] * 7, x, 0, 0, 0, 0, 0, 1, 0.25], dtype=np.float32)
        return {"observation.state": state}

    adapter.append_observation(observation(1))
    np.testing.assert_allclose(adapter.build_batch()["observation.state"][0, :, 0], [0, 0])
    adapter.append_observation(observation(2))
    np.testing.assert_allclose(adapter.build_batch()["observation.state"][0, :, 0], [-1, 0])
    adapter.append_observation(observation(4))
    np.testing.assert_allclose(adapter.build_batch()["observation.state"][0, :, 0], [-2, 0])
    adapter.reset()
    assert not adapter._history
    assert adapter.policy.was_reset


def test_online_adapter_normalizes_raw_hwc_images():
    class Policy:
        pass

    adapter = umi.UMIDiffusionInferenceAdapter(
        Policy(),
        umi.UMIWindowConfig(observation_horizon=2, action_horizon=2),
        camera_keys=["observation.images.pika"],
    )
    state = np.asarray([*[0] * 7, 0, 0, 0, 0, 0, 0, 1, 0.25], dtype=np.float32)
    adapter.append_observation(
        {
            "observation.state": state,
            # Raw Pika robot observations use the camera name, while policy
            # features use the standard observation.images.* prefix.
            "pika": np.full((4, 5, 3), 255, dtype=np.uint8),
        }
    )
    image = adapter.build_batch()["observation.images.pika"]
    assert tuple(image.shape) == (1, 2, 3, 4, 5)
    assert image.dtype.is_floating_point
    assert float(image.min()) == float(image.max()) == 1.0


def test_online_adapter_executes_first_strictly_future_prediction():
    import torch

    class Diffusion:
        @staticmethod
        def generate_actions(batch):
            assert tuple(batch["observation.state"].shape) == (1, 2, 7)
            return torch.tensor([[[1.0] * 7, [2.0] * 7]])

    class Config:
        image_features = {}

    class Policy:
        config = Config()
        diffusion = Diffusion()

        def eval(self):
            return self

    adapter = umi.UMIDiffusionInferenceAdapter(
        Policy(), umi.UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )
    state = np.asarray([*[0] * 7, 0, 0, 0, 0, 0, 0, 1, 0.25], dtype=np.float32)
    action = adapter.predict_relative_action({"observation.state": state})
    np.testing.assert_allclose(action.numpy(), np.ones((1, 7)))


def test_relative_action_decoding_composes_with_current_tcp():
    decoded = umi.UMIDiffusionInferenceAdapter.decode_action(
        [2, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, np.pi / 2, 1.5]
    )
    np.testing.assert_allclose(decoded["tcp_pose"][:3], [3, 0, 0], atol=1e-7)
    np.testing.assert_allclose(
        geometry.rotation_matrix_to_rotvec(geometry.pose_to_matrix(decoded["tcp_pose"])[:3, :3]),
        [0, 0, np.pi / 2],
        atol=1e-7,
    )
    assert decoded["gripper"] == 1.0
