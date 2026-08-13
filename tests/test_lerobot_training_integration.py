"""Integration checks for the FastUMI view and LeRobot's training contract.

The first test runs with the lightweight test dependencies.  The policy smoke
tests are skipped unless the pinned LeRobot package is installed; they exercise
the real policy feature inference and model forward/backward paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot_robot_ufactory.pika_direct import (
    UMIDiffusionRelativeDataset,
    UMIRelativeDataset,
    UMIWindowConfig,
)


class FakeLeRobotDataset:
    def __init__(self, episode_length: int = 8, image_size: int = 64):
        self.meta = SimpleNamespace(
            features={
                "observation.state": {"dtype": "float32", "shape": (18,)},
                "observation.images.pika": {
                    "dtype": "video",
                    "shape": (3, image_size, image_size),
                    "names": ["channels", "height", "width"],
                },
                "action": {"dtype": "float32", "shape": (8,)},
            },
            stats={
                "observation.images.pika": {
                    "mean": np.zeros((3, 1, 1), dtype=np.float32),
                    "std": np.ones((3, 1, 1), dtype=np.float32),
                    "min": np.zeros((3, 1, 1), dtype=np.float32),
                    "max": np.ones((3, 1, 1), dtype=np.float32),
                    "q01": np.zeros((3, 1, 1), dtype=np.float32),
                    "q99": np.ones((3, 1, 1), dtype=np.float32),
                }
            },
            camera_keys=["observation.images.pika"],
            has_language_columns=False,
        )
        self.samples = []
        for frame in range(episode_length):
            tracker = [0, 0, 0, 0, 0, 0, 1]
            tcp = [frame / 100, 0, 0, 0, 0, 0, 1]
            state = np.asarray(
                [*tracker, *tcp, frame / 10, 50, frame / 30, frame], dtype=np.float32
            )
            self.samples.append(
                {
                    "observation.state": state,
                    # LeRobot's policy-side convention after image preprocessing.
                    "observation.images.pika": np.zeros(
                        (3, image_size, image_size), dtype=np.float32
                    ),
                    # The official LeRobot dataset supplies this mask for
                    # chunk policies.  FastUMI only indexes complete windows.
                    "action_is_pad": np.zeros(2, dtype=bool),
                    "episode_index": frame * 0,
                    "frame_index": frame,
                    "task": "integration test",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FakeDiffusionDeltaDataset(FakeLeRobotDataset):
    """Expose the deltas produced by n_obs_steps=2, temporary horizon=4."""

    def __init__(self, episode_length: int = 8, image_size: int = 64):
        super().__init__(episode_length, image_size)
        self.meta.episodes = {"dataset_from_index": [0], "dataset_to_index": [episode_length]}
        self.hf_dataset = []
        for frame, sample in enumerate(self.samples):
            tcp = sample["observation.state"][7:14]
            self.hf_dataset.append(
                {
                    "observation.state": sample["observation.state"],
                    "action": np.asarray([*tcp, sample["observation.state"][14]], dtype=np.float32),
                    "episode_index": np.asarray(0),
                    "frame_index": np.asarray(frame),
                }
            )
        self.num_episodes = 1
        self.episodes = None
        self.absolute_to_relative_idx = None

    def __getitem__(self, index):
        last = len(self.samples) - 1
        obs_indices = [max(0, index - 1), index]
        action_indices = [max(0, min(last, index + delta)) for delta in (-1, 0, 1, 2)]
        return {
            "observation.state": np.stack(
                [self.samples[i]["observation.state"] for i in obs_indices]
            ),
            "observation.images.pika": np.stack(
                [self.samples[i]["observation.images.pika"] for i in obs_indices]
            ),
            "action": np.stack([self.hf_dataset[i]["action"] for i in action_indices]),
            "action_is_pad": np.asarray(
                [index + delta < 0 or index + delta > last for delta in (-1, 0, 1, 2)]
            ),
            "episode_index": np.asarray(0),
            "frame_index": np.asarray(index),
            "task": "integration test",
        }


def test_dataloader_treats_umi_history_as_one_state_feature():
    torch = pytest.importorskip("torch")
    wrapped = UMIRelativeDataset(
        FakeLeRobotDataset(), UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )
    batch = next(iter(torch.utils.data.DataLoader(wrapped, batch_size=2)))

    # History is encoded inside the feature dimension, not exposed as a second
    # observation-time axis.  The action chunk intentionally keeps its time axis.
    assert batch["observation.state"].shape == (2, 14)
    assert batch["action"].shape == (2, 2, 7)
    assert wrapped.meta.features["observation.state"]["shape"] == (14,)
    assert wrapped.meta.features["action"]["shape"] == (7,)


@pytest.mark.parametrize("policy_type", ["act", "diffusion"])
def test_real_lerobot_policy_accepts_umi_batch_and_backpropagates(policy_type):
    """Run only in the target environment containing LeRobot 0.4.3.

    This catches metadata inference, batch rank, normalizer, and model-loss
    incompatibilities that isolated geometry tests cannot detect.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

    window = UMIWindowConfig(observation_horizon=2, action_horizon=2)
    wrapped = (
        UMIRelativeDataset(FakeLeRobotDataset(), window)
        if policy_type == "act"
        else UMIDiffusionRelativeDataset(FakeDiffusionDeltaDataset(), window)
    )
    kwargs = {
        "device": "cpu",
        "n_obs_steps": 1 if policy_type == "act" else 2,
        "n_action_steps": 2,
        "pretrained_backbone_weights": None,
    }
    if policy_type == "act":
        kwargs.update(chunk_size=2, use_vae=False, dim_model=64, n_heads=4, dim_feedforward=128)
    else:
        # One downsampling stage keeps this CPU smoke test small and requires
        # the two-step horizon to be divisible by two.
        kwargs.update(
            horizon=2,
            down_dims=(32,),
            diffusion_step_embed_dim=32,
            n_groups=4,
            do_mask_loss_for_padding=True,
            crop_shape=(64,64),
        )

    policy_cfg = make_policy_config(policy_type, **kwargs)
    policy = make_policy(cfg=policy_cfg, ds_meta=wrapped.meta)
    preprocessor, _ = make_pre_post_processors(policy_cfg, dataset_stats=wrapped.meta.stats)
    batch = next(iter(torch.utils.data.DataLoader(wrapped, batch_size=2)))
    batch = preprocessor(batch)
    loss, _ = policy.forward(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters() if parameter.requires_grad)
