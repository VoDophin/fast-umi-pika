"""Small, dependency-free SE(3) helpers.

Poses use metres and quaternions in ``xyzw`` order.  Relative transforms are
always computed by matrix composition, never by subtracting Euler angles.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion_xyzw) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion is invalid")
    q = q / norm
    return -q if q[3] < 0 else q


def quaternion_to_matrix(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)[:3, :3]
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    # Eigenvector of Davenport's symmetric matrix, robust close to pi.
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
            [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
            [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r.trace()],
        ]
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    return normalize_quaternion(vectors[:, int(np.argmax(values))])


def pose_to_matrix(pose_xyz_xyzw) -> np.ndarray:
    pose = np.asarray(pose_xyz_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose[:3])):
        raise ValueError("pose must be seven finite xyz+xyzw values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    return np.concatenate((transform[:3, 3], matrix_to_quaternion(transform[:3, :3])))


def rotation_matrix_to_rotvec(rotation) -> np.ndarray:
    q = matrix_to_quaternion(rotation)
    vector_norm = float(np.linalg.norm(q[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, q[3])
    return q[:3] * (angle / vector_norm)


def rotvec_to_rotation_matrix(rotation_vector) -> np.ndarray:
    """Convert an axis-angle rotation vector to a 3x3 rotation matrix."""
    rotvec = np.asarray(rotation_vector, dtype=np.float64)
    if rotvec.shape != (3,) or not np.all(np.isfinite(rotvec)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def relative_rotvec_pose_to_absolute(reference_pose, relative_xyz_rotvec) -> np.ndarray:
    """Apply a relative xyz+rotation-vector transform to an absolute pose."""
    relative_pose_value = np.asarray(relative_xyz_rotvec, dtype=np.float64)
    if relative_pose_value.shape != (6,) or not np.all(np.isfinite(relative_pose_value)):
        raise ValueError("relative pose must contain six finite xyz+rotvec values")
    relative_transform = np.eye(4, dtype=np.float64)
    relative_transform[:3, :3] = rotvec_to_rotation_matrix(relative_pose_value[3:])
    relative_transform[:3, 3] = relative_pose_value[:3]
    return matrix_to_pose(pose_to_matrix(reference_pose) @ relative_transform)


def relative_pose(reference_pose, target_pose, representation="xyz_rotvec") -> np.ndarray:
    relative = np.linalg.inv(pose_to_matrix(reference_pose)) @ pose_to_matrix(target_pose)
    if representation == "xyz_xyzw":
        return matrix_to_pose(relative)
    if representation == "xyz_rotvec":
        return np.concatenate((relative[:3, 3], rotation_matrix_to_rotvec(relative[:3, :3])))
    raise ValueError(f"unsupported pose representation: {representation}")


def compose_pose(parent_pose, child_pose) -> np.ndarray:
    return matrix_to_pose(pose_to_matrix(parent_pose) @ pose_to_matrix(child_pose))
