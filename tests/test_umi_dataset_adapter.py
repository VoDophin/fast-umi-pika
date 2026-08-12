import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src/lerobot_robot_ufactory/pika_direct"
package = types.ModuleType("pika_adapter_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for module_name in ("geometry", "umi_relative"):
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.{module_name}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
umi = sys.modules["pika_adapter_test.umi_relative"]


class FakeLeRobotDataset:
    def __init__(self):
        class Meta:
            features = {
                "observation.state": {"dtype": "float32", "shape": (18,)},
                "action": {"dtype": "float32", "shape": (8,)},
            }
            stats = {}
            camera_keys = ["observation.images.pika"]
            has_language_columns = False
        self.meta = Meta()
        self.samples = []
        for episode, length in ((0, 6), (1, 5)):
            for frame in range(length):
                tracker = [0, 0, 0, 0, 0, 0, 1]
                tcp = [frame, 0, 0, 0, 0, 0, 1]
                state = np.asarray([*tracker, *tcp, frame / 10, 50, frame / 30, frame], dtype=np.float32)
                self.samples.append({
                    "observation.state": state,
                    "observation.images.pika": np.zeros((4, 4, 3), dtype=np.uint8),
                    "episode_index": episode,
                    "frame_index": frame,
                    "task": "test",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_adapter_preserves_lerobot_fields_and_episode_boundaries():
    wrapped = umi.UMIRelativeDataset(
        FakeLeRobotDataset(), umi.UMIWindowConfig(observation_horizon=2, action_horizon=2)
    )
    assert len(wrapped) == (6 - 2 - 2 + 1) + (5 - 2 - 2 + 1)
    item = wrapped[0]
    assert item["observation.state"].shape == (2 * 6 + 2,)
    assert item["action"].shape == (2, 7)
    assert item["observation.images.pika"].shape == (4, 4, 3)
    assert item["task"] == "test"
    np.testing.assert_allclose(item["action"][:, 0], [1, 2])
    assert wrapped.meta.features["observation.state"]["shape"] == (14,)
    assert wrapped.meta.features["action"]["shape"] == (7,)
    assert wrapped.num_frames == len(wrapped)
    assert wrapped.meta.episodes["dataset_from_index"] == [0, 3]
    assert "mean" in wrapped.meta.stats["action"]
