import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src/lerobot_robot_ufactory/pika_direct"
package = types.ModuleType("pika_direct_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for module_name in ("geometry", "umi_relative"):
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.{module_name}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
processor_module = sys.modules["pika_direct_test.umi_relative"]


def test_umi_window_uses_one_current_reference():
    poses = np.array([[i, 0, 0, 0, 0, 0, 1] for i in range(6)], dtype=float)
    gripper = np.linspace(0, 1, 6)
    processor = processor_module.UMIRelativeEEProcessor(
        processor_module.UMIWindowConfig(observation_horizon=3, action_horizon=2)
    )
    result = processor.process_window(poses, gripper, current_index=2)
    np.testing.assert_allclose(result["observation.relative_tcp"][:, 0], [-2, -1, 0])
    np.testing.assert_allclose(result["observation.relative_tcp"][-1], np.zeros(6), atol=1e-8)
    np.testing.assert_allclose(result["action"][:, 0], [1, 2])
    np.testing.assert_allclose(result["action"][:, -1], gripper[3:5])


def test_only_complete_windows_are_indexed():
    processor = processor_module.UMIRelativeEEProcessor(
        processor_module.UMIWindowConfig(observation_horizon=3, action_horizon=2)
    )
    assert list(processor.valid_current_indices(6)) == [2, 3]
    assert list(processor.valid_current_indices(4)) == []
