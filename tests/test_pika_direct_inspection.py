import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src/lerobot_robot_ufactory/pika_direct"
package = types.ModuleType("pika_inspection_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for module_name in ("geometry", "umi_relative", "inspection"):
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.{module_name}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
inspection = sys.modules["pika_inspection_test.inspection"]


def test_inspector_reports_ranges_and_clean_sequence():
    samples = []
    for frame in range(3):
        tracker = [frame * 0.01, 0, 0, 0, 0, 0, 1]
        tcp = [frame * 0.01, 0, 0.1, 0, 0, 0, 1]
        state = np.asarray([*tracker, *tcp, frame / 2, 50, 10 + frame / 30, frame + 1])
        samples.append({
            "observation.state": state,
            "observation.images.pika": np.zeros((8, 8, 3), dtype=np.uint8),
            "frame_index": frame,
        })
    report = inspection.inspect_samples(samples)
    assert report.frame_count == 3
    assert report.image_shapes == {"pika": (8, 8, 3)}
    assert report.errors == []
    np.testing.assert_allclose(report.fps_estimate, 30, rtol=1e-5)
