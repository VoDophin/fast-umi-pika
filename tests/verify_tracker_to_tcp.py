"""Verify the configured Tracker-to-TCP extrinsic with live Pika poses.

Usage from the repository root::

    python tests/verify_tracker_to_tcp.py \
        --config config/pika/pika_direct_record.yaml

During sampling, keep the physical TCP pressed against one fixed point while
rotating the Pika around it.  A useful extrinsic makes the calculated TCP
position move substantially less than the raw Tracker position.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot_robot_ufactory.devices.pika import PikaDevice
from lerobot_robot_ufactory.pika_direct.geometry import compose_pose, normalize_quaternion


def load_hardware_config(path: Path) -> tuple[str, str, np.ndarray]:
    """Read only the Pika port, Tracker ID and extrinsic from recorder YAML."""
    with path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)

    robot = document.get("robot", {})
    port = robot.get("port")
    tracker_id = robot.get("tracker_device_id")
    extrinsic = robot.get("tracker_to_tcp", {})
    translation = extrinsic.get("translation_m", [0.0, 0.0, 0.0])
    rotation = extrinsic.get("rotation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])

    if not port:
        raise ValueError(f"robot.port is missing in {path}")
    if not tracker_id:
        raise ValueError(f"robot.tracker_device_id is missing in {path}")

    tracker_to_tcp = np.asarray([*translation, *rotation], dtype=np.float64)
    if tracker_to_tcp.shape != (7,) or not np.all(np.isfinite(tracker_to_tcp)):
        raise ValueError("tracker_to_tcp must contain 3 translation and 4 finite quaternion values")
    tracker_to_tcp[3:] = normalize_quaternion(tracker_to_tcp[3:])
    return str(port), str(tracker_id), tracker_to_tcp


def read_valid_pose(sense, tracker_id: str) -> np.ndarray | None:
    """Return one normalized xyz+xyzw Tracker pose, or None for an invalid read."""
    pose = sense.get_pose(tracker_id)
    if pose is None:
        return None
    value = np.asarray([*pose.position, *pose.rotation], dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        return None
    try:
        value[3:] = normalize_quaternion(value[3:])
    except ValueError:
        return None
    return value


def position_statistics(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return mean, peak-to-peak range, RMS spread and maximum spread in metres."""
    center = positions.mean(axis=0)
    offsets = np.linalg.norm(positions - center, axis=1)
    rms = float(np.sqrt(np.mean(offsets**2)))
    return center, np.ptp(positions, axis=0), rms, float(offsets.max())


def format_xyz_metres(value: np.ndarray) -> str:
    return "[" + ", ".join(f"{component:+.5f}" for component in value) + "] m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定实际 TCP 并旋转 Pika，比较原始 Tracker 与外参转换后 TCP 的位置漂移。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pika/pika_direct_record.yaml"),
        help="采集 YAML 配置路径",
    )
    parser.add_argument("--samples", type=int, default=300, help="有效采样数量")
    parser.add_argument("--interval", type=float, default=0.02, help="采样间隔，单位秒")
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.interval < 0:
        parser.error("--interval must not be negative")

    port, tracker_id, tracker_to_tcp = load_hardware_config(args.config)
    is_identity = np.allclose(tracker_to_tcp[:3], 0.0) and np.allclose(
        tracker_to_tcp[3:], [0.0, 0.0, 0.0, 1.0]
    )

    print("读取的配置：")
    print(f"  Sense port:       {port}")
    print(f"  Tracker ID:       {tracker_id}")
    print(f"  translation:      {format_xyz_metres(tracker_to_tcp[:3])}")
    print(f"  rotation (xyzw):  {tracker_to_tcp[3:].round(7).tolist()}")
    if is_identity:
        print("\n警告：当前是单位外参，计算出的 TCP 将与 Tracker 原点完全相同。")

    device = PikaDevice(
        1,
        pika_sense_port=port,
        pika_tracker_device=tracker_id,
    )
    sense = device.pika_sense

    try:
        input(
            "\n把定义好的实际 TCP 抵在一个固定点上。采样期间保持该点不动，"
            "并尽量改变 Pika 姿态；准备好后按 Enter："
        )
        tracker_samples: list[np.ndarray] = []
        tcp_samples: list[np.ndarray] = []
        invalid_reads = 0

        while len(tracker_samples) < args.samples:
            tracker_pose = read_valid_pose(sense, tracker_id)
            if tracker_pose is None:
                invalid_reads += 1
                time.sleep(args.interval)
                continue

            tcp_pose = compose_pose(tracker_pose, tracker_to_tcp)
            tracker_samples.append(tracker_pose)
            tcp_samples.append(tcp_pose)
            print(
                f"\r有效采样：{len(tracker_samples):4d}/{args.samples}，"
                f"无效读取：{invalid_reads}",
                end="",
                flush=True,
            )
            time.sleep(args.interval)
        print()

        tracker_array = np.stack(tracker_samples)
        tcp_array = np.stack(tcp_samples)
        tracker_center, tracker_range, tracker_rms, tracker_max = position_statistics(
            tracker_array[:, :3]
        )
        tcp_center, tcp_range, tcp_rms, tcp_max = position_statistics(tcp_array[:, :3])
        ratio = tcp_rms / tracker_rms if tracker_rms > 1e-12 else math.inf

        print("\n第一帧对比：")
        print(f"  Tracker xyz: {format_xyz_metres(tracker_array[0, :3])}")
        print(f"  TCP xyz:     {format_xyz_metres(tcp_array[0, :3])}")
        print("\n固定点旋转统计：")
        print(f"  Tracker xyz峰峰值: {format_xyz_metres(tracker_range)}")
        print(f"  TCP xyz峰峰值:     {format_xyz_metres(tcp_range)}")
        print(f"  Tracker RMS漂移:   {tracker_rms * 1000:.2f} mm")
        print(f"  TCP RMS漂移:       {tcp_rms * 1000:.2f} mm")
        print(f"  Tracker最大漂移:   {tracker_max * 1000:.2f} mm")
        print(f"  TCP最大漂移:       {tcp_max * 1000:.2f} mm")
        print(f"  TCP/Tracker RMS比: {ratio:.3f}")
        print(f"  Tracker均值:       {format_xyz_metres(tracker_center)}")
        print(f"  TCP均值:           {format_xyz_metres(tcp_center)}")

        print("\n解释：")
        if is_identity:
            print("  当前为单位外参，因此两个漂移必然相同；这只能证明代码读取正常，不能证明 TCP 正确。")
        elif tracker_rms < 0.001:
            print("  Tracker移动不足 1 mm；姿态变化可能太小，请扩大旋转角度后重新测试。")
        elif ratio < 0.5:
            print("  转换后的 TCP 漂移明显小于 Tracker，translation 外参方向基本合理。")
        else:
            print("  TCP漂移没有明显减小；请检查固定点操作和 translation 外参的方向、正负号及长度。")
        print("  本测试主要验证平移外参；旋转外参还需结合实际工具坐标轴方向检查。")

    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
