import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).parents[1]
    / "src/lerobot_robot_ufactory/pika_direct/sync_inspection.py"
)
spec = importlib.util.spec_from_file_location("pika_sync_inspection_test", PATH)
sync = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sync
spec.loader.exec_module(sync)


def _sample(frame: int, timestamp: float, image_value: int, tcp_x: float) -> dict:
    tracker = [tcp_x, 0, 0, 0, 0, 0, 1]
    tcp = [tcp_x, 0, 0, 0, 0, 0, 1]
    state = np.asarray([*tracker, *tcp, 0.5, 50.0, timestamp, frame + 1])
    return {
        "observation.state": state,
        "observation.images.pika": np.full((8, 8, 3), image_value, dtype=np.uint8),
        "frame_index": frame,
        "episode_index": 0,
    }


def test_sync_report_detects_gap_and_frozen_image_while_tcp_moves():
    timestamps = [0.0, 1 / 30, 2 / 30, 5 / 30]
    samples = [
        _sample(frame, timestamp, image_value=0, tcp_x=frame * 0.01)
        for frame, timestamp in enumerate(timestamps)
    ]
    report = sync.inspect_sync(samples, target_fps=30)

    assert report.long_gap_count == 1
    assert report.frozen_image_pair_count == 3
    assert report.frozen_while_tcp_moving_count == 3
    assert any("frozen" in warning for warning in report.warnings)
    assert sync.LIMITATION in report.limitations


def test_lag_sign_convention_finds_image_signal_that_lags_tcp():
    tcp_motion = np.asarray([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6], dtype=float)
    # Image transition i shows the TCP transition from two frames earlier.
    image_motion = np.concatenate((np.zeros(2), tcp_motion[:-2]))
    correlations = sync._lag_correlations(image_motion, tcp_motion, max_lag_frames=3)

    assert max(correlations, key=correlations.get) == -2

