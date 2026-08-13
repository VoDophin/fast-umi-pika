"""Inspect temporal quality of a recorded Pika episode.

IMPORTANT LIMITATION
--------------------
The current raw format stores one local ``sensor_timestamp`` after reading the
Tracker and gripper and before reading the camera. It does not store the
camera exposure timestamp, Tracker timestamp, and gripper timestamp
separately. Therefore this module can detect timing irregularities and
estimate a *possible* frame lag from motion correlation, but it cannot measure
the true hardware synchronization error in milliseconds.

Run from the project environment:

    python -m lerobot_robot_ufactory.pika_direct.sync_inspection \
        --repo-id local/pika_direct \
        --root ./data/pika_direct \
        --episode 0 \
        --fps 30 \
        --json-output ./sync_report_episode_0.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


LIMITATION = (
    "Raw frames do not contain separate camera/Tracker/gripper hardware timestamps; "
    "reported lag is a motion-correlation estimate, not an exact synchronization measurement."
)


@dataclass
class SyncInspectionReport:
    episode_index: int
    camera_key: str
    frame_count: int
    target_fps: float
    measured_fps: float | None
    duration_s: float
    dt_median_ms: float | None
    dt_p95_ms: float | None
    jitter_p95_ms: float | None
    long_gap_count: int
    short_gap_count: int
    sample_id_gap_count: int
    duplicate_frame_index_count: int
    frozen_image_pair_count: int
    frozen_image_pair_fraction: float
    frozen_while_tcp_moving_count: int
    tcp_moving_pair_count: int
    zero_lag_correlation: float | None
    best_lag_frames: int | None
    best_lag_ms: float | None
    best_lag_correlation: float | None
    lag_interpretation: str
    limitations: list[str] = field(default_factory=lambda: [LIMITATION])
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_array(value).reshape(-1)[0])


def _camera_keys(sample: dict[str, Any]) -> list[str]:
    return sorted(key for key in sample if key.startswith("observation.images."))


def _prepare_gray(image: Any, max_side: int = 160) -> np.ndarray:
    """Convert HWC/CHW uint8 or float image to a small float32 grayscale image."""
    value = _array(image)
    if value.ndim != 3:
        raise ValueError(f"camera image must be rank 3, got shape {value.shape}")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"camera image must be HWC or CHW, got shape {value.shape}")
    value = value.astype(np.float32, copy=False)
    if value.size and float(np.nanmax(value)) > 1.5:
        value = value / 255.0
    if value.shape[-1] == 1:
        gray = value[..., 0]
    else:
        # Exact color order is irrelevant for a frame-difference diagnostic.
        gray = value[..., :3].mean(axis=-1)
    step = max(1, int(np.ceil(max(gray.shape) / max_side)))
    return np.ascontiguousarray(gray[::step, ::step], dtype=np.float32)


def _quaternion_step_angles(quaternions: np.ndarray) -> np.ndarray:
    normalized = quaternions.astype(np.float64, copy=True)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("TCP contains a zero quaternion")
    normalized /= norms
    dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
    # q and -q represent the same rotation.
    return 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))


def _normalized_motion(values: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(values, 90)) if len(values) else 0.0
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / scale


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 8 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _lag_correlations(
    image_motion: np.ndarray, tcp_motion: np.ndarray, max_lag_frames: int
) -> dict[int, float]:
    """Correlate image_motion[i] with tcp_motion[i + lag].

    Positive lag means the image signal best matches a later TCP transition
    (image appears to lead TCP). Negative lag means it best matches an earlier
    TCP transition (image appears to lag TCP).
    """
    result: dict[int, float] = {}
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            x, y = image_motion[-lag:], tcp_motion[:lag]
        elif lag > 0:
            x, y = image_motion[:-lag], tcp_motion[lag:]
        else:
            x, y = image_motion, tcp_motion
        value = _correlation(x, y)
        if value is not None:
            result[lag] = value
    return result


def inspect_sync(
    samples: Sequence[dict[str, Any]],
    *,
    episode_index: int = 0,
    camera_key: str | None = None,
    target_fps: float = 30.0,
    max_lag_frames: int = 5,
    frozen_mae_threshold: float = 1e-4,
    moving_translation_threshold_m: float = 1e-3,
    moving_rotation_threshold_rad: float = 1e-2,
) -> SyncInspectionReport:
    """Inspect one episode without modifying it."""
    if len(samples) < 2:
        raise ValueError("sync inspection requires at least two frames")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    available_cameras = _camera_keys(samples[0])
    if not available_cameras:
        raise ValueError("episode contains no observation.images.* field")
    if camera_key is None:
        camera_key = available_cameras[0]
    elif not camera_key.startswith("observation.images."):
        camera_key = f"observation.images.{camera_key}"
    if camera_key not in available_cameras:
        raise ValueError(f"camera {camera_key!r} not found; available: {available_cameras}")

    states = np.stack([_array(sample["observation.state"]).reshape(-1) for sample in samples])
    if states.shape[1] < 18:
        raise ValueError("Pika raw observation.state must contain at least 18 values")
    tcp = states[:, 7:14].astype(np.float64)
    timestamps = states[:, 16].astype(np.float64)
    sample_ids = states[:, 17].astype(np.float64)
    frame_indices = np.asarray([_scalar(sample["frame_index"]) for sample in samples])
    images = [_prepare_gray(sample[camera_key]) for sample in samples]
    if any(image.shape != images[0].shape for image in images[1:]):
        raise ValueError("camera image shape changes within the episode")

    errors: list[str] = []
    warnings: list[str] = []
    if not all(np.all(np.isfinite(value)) for value in (tcp, timestamps, sample_ids)):
        errors.append("NaN or infinite value in TCP/timestamp/sample_id")

    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        errors.append("sensor_timestamp is duplicate or non-monotonic")
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    target_dt = 1.0 / target_fps
    long_gap_count = int(np.count_nonzero(valid_dt > 1.5 * target_dt))
    short_gap_count = int(np.count_nonzero(valid_dt < 0.5 * target_dt))
    if long_gap_count:
        warnings.append(f"{long_gap_count} frame intervals exceed 1.5x the target interval")
    if short_gap_count:
        warnings.append(f"{short_gap_count} frame intervals are below 0.5x the target interval")

    sample_steps = np.diff(sample_ids)
    sample_id_gap_count = int(np.count_nonzero(sample_steps != 1))
    if sample_id_gap_count:
        warnings.append(f"{sample_id_gap_count} sample_id gaps detected")
    frame_steps = np.diff(frame_indices)
    duplicate_frame_index_count = int(np.count_nonzero(frame_steps == 0))
    if duplicate_frame_index_count:
        errors.append(f"{duplicate_frame_index_count} duplicate frame_index values detected")
    if np.any(frame_steps != 1):
        errors.append("frame_index is not a contiguous increasing sequence")

    image_motion = np.asarray(
        [float(np.mean(np.abs(images[i] - images[i - 1]))) for i in range(1, len(images))]
    )
    translation_step = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    try:
        rotation_step = _quaternion_step_angles(tcp[:, 3:7])
    except ValueError as exc:
        errors.append(str(exc))
        rotation_step = np.zeros_like(translation_step)
    tcp_moving = (translation_step >= moving_translation_threshold_m) | (
        rotation_step >= moving_rotation_threshold_rad
    )
    frozen = image_motion <= frozen_mae_threshold
    frozen_while_moving = frozen & tcp_moving
    frozen_image_pair_count = int(np.count_nonzero(frozen))
    frozen_while_moving_count = int(np.count_nonzero(frozen_while_moving))
    tcp_moving_pair_count = int(np.count_nonzero(tcp_moving))
    if frozen_while_moving_count:
        warnings.append(
            f"image is frozen for {frozen_while_moving_count} intervals while TCP is moving"
        )

    tcp_motion = _normalized_motion(translation_step) + _normalized_motion(rotation_step)
    correlations = _lag_correlations(image_motion, tcp_motion, max_lag_frames)
    zero_lag = correlations.get(0)
    best_lag = max(correlations, key=correlations.get) if correlations else None
    best_correlation = correlations.get(best_lag) if best_lag is not None else None
    median_dt = float(np.median(valid_dt)) if len(valid_dt) else None
    best_lag_ms = (
        float(best_lag * median_dt * 1000.0)
        if best_lag is not None and median_dt is not None
        else None
    )
    if best_lag is None:
        lag_interpretation = "inconclusive: insufficient varying motion"
    elif best_lag == 0:
        lag_interpretation = "best motion correlation occurs at zero frame offset"
    elif best_lag > 0:
        lag_interpretation = "image motion appears to lead TCP motion"
    else:
        lag_interpretation = "image motion appears to lag TCP motion"

    # Only warn when the non-zero peak is meaningful and improves on zero lag.
    zero_for_comparison = zero_lag if zero_lag is not None else -1.0
    if (
        best_lag not in (None, 0)
        and best_correlation is not None
        and best_correlation >= 0.35
        and best_correlation - zero_for_comparison >= 0.08
    ):
        warnings.append(
            f"possible image/TCP offset: best lag {best_lag:+d} frames "
            f"({best_lag_ms:+.1f} ms), correlation={best_correlation:.3f}"
        )

    dt_median_ms = median_dt * 1000.0 if median_dt is not None else None
    dt_p95_ms = float(np.percentile(valid_dt, 95) * 1000.0) if len(valid_dt) else None
    jitter_p95_ms = (
        float(np.percentile(np.abs(valid_dt - median_dt), 95) * 1000.0)
        if median_dt is not None
        else None
    )
    measured_fps = 1.0 / median_dt if median_dt is not None else None
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0
    return SyncInspectionReport(
        episode_index=episode_index,
        camera_key=camera_key,
        frame_count=len(samples),
        target_fps=float(target_fps),
        measured_fps=measured_fps,
        duration_s=duration_s,
        dt_median_ms=dt_median_ms,
        dt_p95_ms=dt_p95_ms,
        jitter_p95_ms=jitter_p95_ms,
        long_gap_count=long_gap_count,
        short_gap_count=short_gap_count,
        sample_id_gap_count=sample_id_gap_count,
        duplicate_frame_index_count=duplicate_frame_index_count,
        frozen_image_pair_count=frozen_image_pair_count,
        frozen_image_pair_fraction=frozen_image_pair_count / len(image_motion),
        frozen_while_tcp_moving_count=frozen_while_moving_count,
        tcp_moving_pair_count=tcp_moving_pair_count,
        zero_lag_correlation=zero_lag,
        best_lag_frames=best_lag,
        best_lag_ms=best_lag_ms,
        best_lag_correlation=best_correlation,
        lag_interpretation=lag_interpretation,
        errors=errors,
        warnings=warnings,
    )


def _format_optional(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def print_report(report: SyncInspectionReport) -> None:
    print("=== Pika capture timing quality report ===")
    print(f"episode:                 {report.episode_index}")
    print(f"camera:                  {report.camera_key}")
    print(f"frames:                  {report.frame_count}")
    print(f"duration:                {report.duration_s:.3f} s")
    print(f"target / measured FPS:   {report.target_fps:.3f} / {_format_optional(report.measured_fps)}")
    print(f"median / p95 dt:         {_format_optional(report.dt_median_ms)} / {_format_optional(report.dt_p95_ms)} ms")
    print(f"p95 interval jitter:     {_format_optional(report.jitter_p95_ms)} ms")
    print(f"long / short intervals:  {report.long_gap_count} / {report.short_gap_count}")
    print(f"sample-id gaps:          {report.sample_id_gap_count}")
    print(
        "frozen image pairs:     "
        f"{report.frozen_image_pair_count} ({report.frozen_image_pair_fraction:.1%})"
    )
    print(
        "frozen while moving:    "
        f"{report.frozen_while_tcp_moving_count}/{report.tcp_moving_pair_count} TCP-moving pairs"
    )
    print(f"zero-lag correlation:    {_format_optional(report.zero_lag_correlation)}")
    print(
        "best lag:                "
        f"{report.best_lag_frames if report.best_lag_frames is not None else 'N/A'} frames, "
        f"{_format_optional(report.best_lag_ms, 1)} ms, "
        f"corr={_format_optional(report.best_lag_correlation)}"
    )
    print(f"lag interpretation:      {report.lag_interpretation}")
    if report.errors:
        print("ERRORS:")
        for message in report.errors:
            print(f"  - {message}")
    if report.warnings:
        print("WARNINGS:")
        for message in report.warnings:
            print(f"  - {message}")
    print("LIMITATION:")
    for message in report.limitations:
        print(f"  - {message}")


def _load_episode(dataset: Any, episode_index: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if int(_scalar(sample["episode_index"])) == episode_index:
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect timing irregularities and possible image/TCP lag in one Pika episode"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", help="camera name or full observation.images.* key")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-lag-frames", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    samples = _load_episode(dataset, args.episode)
    report = inspect_sync(
        samples,
        episode_index=args.episode,
        camera_key=args.camera,
        target_fps=args.fps,
        max_lag_frames=args.max_lag_frames,
    )
    print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON report written to: {args.json_output}")


if __name__ == "__main__":
    main()
