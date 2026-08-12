import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "src/lerobot_robot_ufactory/pika_direct/geometry.py"
SPEC = importlib.util.spec_from_file_location("pika_direct_geometry", PATH)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def pose(x=0, y=0, z=0, q=(0, 0, 0, 1)):
    return np.array([x, y, z, *q], dtype=float)


def test_identity_and_pure_translation():
    np.testing.assert_allclose(geometry.relative_pose(pose(), pose()), np.zeros(6), atol=1e-8)
    np.testing.assert_allclose(geometry.relative_pose(pose(), pose(1, 2, 3))[:3], [1, 2, 3])


def test_pure_rotation_and_composition():
    half = np.sqrt(0.5)
    result = geometry.relative_pose(pose(), pose(q=(0, 0, half, half)))
    np.testing.assert_allclose(result[3:], [0, 0, np.pi / 2], atol=1e-7)
    tcp = geometry.compose_pose(pose(1, 0, 0, q=(0, 0, half, half)), pose(1, 0, 0))
    np.testing.assert_allclose(tcp[:3], [1, 1, 0], atol=1e-7)


def test_global_rigid_transform_invariance():
    half = np.sqrt(0.5)
    global_pose = pose(4, -2, 1, (0, 0, half, half))
    reference = pose(1, 2, 3)
    target = pose(2, 4, 6, (half, 0, 0, half))
    expected = geometry.relative_pose(reference, target)
    transformed_reference = geometry.compose_pose(global_pose, reference)
    transformed_target = geometry.compose_pose(global_pose, target)
    np.testing.assert_allclose(
        geometry.relative_pose(transformed_reference, transformed_target), expected, atol=1e-7
    )


def test_quaternion_is_normalized_and_zero_rejected():
    np.testing.assert_allclose(geometry.normalize_quaternion([0, 0, 0, 2]), [0, 0, 0, 1])
    try:
        geometry.normalize_quaternion([0, 0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("zero quaternion must fail")
