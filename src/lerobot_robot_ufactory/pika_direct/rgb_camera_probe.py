"""Find RGB-capable V4L2 nodes and save one inspection frame per node."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PREFERRED_RGB_FORMATS = ("MJPG", "JPEG", "RGB3", "BGR3", "YUYV")
OTHER_COLOR_FORMATS = ("UYVY", "NV12", "NV21")
VIDEO_NODE_RE = re.compile(r"^video(\d+)$")
FORMAT_RE = re.compile(r"\[\d+\]:\s+'([^']+)'", re.MULTILINE)


def parse_v4l2_formats(output: str) -> tuple[str, ...]:
    """Extract normalized FOURCC values from ``v4l2-ctl`` output."""
    return tuple(dict.fromkeys(match.strip().upper() for match in FORMAT_RE.findall(output)))


def parse_udev_properties(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def rgb_candidate_rank(formats: tuple[str, ...]) -> tuple[int, str]:
    """Rank formats without mistaking pure depth/greyscale nodes for RGB."""
    preferred = [fmt for fmt in PREFERRED_RGB_FORMATS if fmt in formats]
    if preferred:
        return 2, f"preferred color format: {', '.join(preferred)}"
    fallback = [fmt for fmt in OTHER_COLOR_FORMATS if fmt in formats]
    if fallback:
        return 1, f"possible color format: {', '.join(fallback)}"
    return 0, "no RGB-compatible packed color format"


def video_node_sort_key(path: Path) -> tuple[int, str]:
    match = VIDEO_NODE_RE.match(path.name)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def list_video_nodes(device_root: Path = Path("/dev")) -> list[Path]:
    return sorted(
        (path for path in device_root.glob("video*") if VIDEO_NODE_RE.match(path.name)),
        key=video_node_sort_key,
    )


def aliases_for_node(node: Path, alias_root: Path) -> list[str]:
    if not alias_root.exists():
        return []
    target = node.resolve()
    aliases = []
    for alias in alias_root.iterdir():
        try:
            if alias.resolve() == target:
                aliases.append(str(alias))
        except OSError:
            continue
    return sorted(aliases)


def _run_text(command: list[str]) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return "", f"command not found: {command[0]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        return completed.stdout, message or f"exit status {completed.returncode}"
    return completed.stdout, None


def inspect_node(node: Path) -> dict[str, Any]:
    formats_output, formats_error = _run_text(
        ["v4l2-ctl", "--device", str(node), "--list-formats-ext"]
    )
    properties_output, properties_error = _run_text(
        ["udevadm", "info", "--query=property", f"--name={node}"]
    )
    formats = parse_v4l2_formats(formats_output)
    rank, reason = rgb_candidate_rank(formats)
    return {
        "node": str(node),
        "formats": list(formats),
        "candidate_rank": rank,
        "candidate_reason": reason,
        "by_id": aliases_for_node(node, Path("/dev/v4l/by-id")),
        "by_path": aliases_for_node(node, Path("/dev/v4l/by-path")),
        "udev": parse_udev_properties(properties_output),
        "formats_error": formats_error,
        "properties_error": properties_error,
    }


def preferred_device_path(candidate: dict[str, Any]) -> str:
    if candidate["by_id"]:
        return candidate["by_id"][0]
    if candidate["by_path"]:
        return candidate["by_path"][0]
    return candidate["node"]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "camera"


def should_attempt_capture(candidate: dict[str, Any], exhaustive: bool) -> bool:
    if exhaustive:
        # Empty format output without an error is normally a metadata-only node.
        return bool(candidate["formats"]) or candidate["formats_error"] is not None
    return candidate["candidate_rank"] > 0


def capture_profiles(formats: list[str]) -> list[dict[str, Any]]:
    """Try the device default first, then increasingly explicit profiles."""
    profiles: list[dict[str, Any]] = [
        {"name": "device-default", "configure": False, "fourcc": None},
        {"name": "requested-size-default-fourcc", "configure": True, "fourcc": None},
    ]
    for fourcc in (*PREFERRED_RGB_FORMATS, *OTHER_COLOR_FORMATS, *formats):
        if len(fourcc) != 4 or any(item["fourcc"] == fourcc for item in profiles):
            continue
        profiles.append(
            {"name": f"requested-size-{fourcc}", "configure": True, "fourcc": fourcc}
        )
    return profiles


def capture_frame(
    candidate: dict[str, Any],
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on target environment
        return {"ok": False, "error": f"OpenCV dependency unavailable: {exc}"}

    node = candidate["node"]
    cv2.setNumThreads(1)
    attempts: list[dict[str, Any]] = []
    frame = None
    actual_profile: dict[str, Any] = {}
    successful_profile: dict[str, Any] | None = None

    for profile in capture_profiles(candidate["formats"]):
        capture = cv2.VideoCapture(node, cv2.CAP_V4L2)
        try:
            if not capture.isOpened():
                attempts.append({"profile": profile["name"], "error": "open failed"})
                # FOURCC and resolution are configured only after opening, so
                # repeating profiles cannot recover an open-level failure.
                break
            if profile["configure"]:
                if profile["fourcc"]:
                    capture.set(
                        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*profile["fourcc"])
                    )
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_FPS, fps)

            current_frame = None
            for _ in range(max(1, warmup_frames + 1)):
                ok, value = capture.read()
                if ok and value is not None:
                    current_frame = value
            if current_frame is None:
                attempts.append({"profile": profile["name"], "error": "read failed"})
                continue

            frame = current_frame
            successful_profile = profile
            actual_profile = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            }
            attempts.append({"profile": profile["name"], "ok": True})
            break
        except Exception as exc:  # pragma: no cover - hardware/backend dependent
            attempts.append({"profile": profile["name"], "error": str(exc)})
        finally:
            capture.release()

    if frame is None:
        return {
            "ok": False,
            "error": "all OpenCV capture profiles failed",
            "attempts": attempts,
        }

    try:
        original_shape = tuple(frame.shape)
        if frame.ndim == 2:
            display_frame = frame
        elif frame.ndim == 3 and frame.shape[2] == 1:
            display_frame = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[2] == 2:
            display_frame = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            display_frame = frame[:, :, :3]
        else:
            return {
                "ok": False,
                "error": f"unsupported frame shape: {original_shape}",
                "attempts": attempts,
            }

        if display_frame.dtype != np.uint8:
            minimum = float(np.min(display_frame))
            maximum = float(np.max(display_frame))
            if maximum > minimum:
                display_frame = ((display_frame - minimum) * (255.0 / (maximum - minimum))).astype(
                    np.uint8
                )
            else:
                display_frame = np.zeros(display_frame.shape, dtype=np.uint8)

        serial = candidate["udev"].get("ID_SERIAL_SHORT", "unknown-serial")
        filename = f"{Path(node).name}_{_safe_filename(serial)}.jpg"
        image_path = output_dir / filename
        if not cv2.imwrite(str(image_path), display_frame):
            return {"ok": False, "error": f"failed to write {image_path}", "attempts": attempts}

        if display_frame.ndim == 3 and display_frame.shape[2] >= 3:
            sample = display_frame[:, :, :3].astype(np.int16)
        else:
            sample = None
        if sample is not None:
            channel_difference = float(
                np.mean(
                    (
                        np.abs(sample[:, :, 0] - sample[:, :, 1])
                        + np.abs(sample[:, :, 1] - sample[:, :, 2])
                        + np.abs(sample[:, :, 2] - sample[:, :, 0])
                    )
                    / 3.0
                )
            )
        else:
            channel_difference = 0.0
        return {
            "ok": True,
            "image": str(image_path),
            "shape": list(original_shape),
            "capture_profile": successful_profile["name"],
            "selected_fourcc": successful_profile["fourcc"],
            "actual_width": actual_profile["width"],
            "actual_height": actual_profile["height"],
            "actual_fps": actual_profile["fps"],
            "channel_difference": channel_difference,
            "attempts": attempts,
        }
    except Exception as exc:  # pragma: no cover - hardware/backend dependent
        return {"ok": False, "error": str(exc), "attempts": attempts}


def probe_rgb_cameras(
    output_root: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 5,
    exhaustive: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError("v4l2-ctl is required; install the Linux package 'v4l-utils'")
    if min(width, height, fps) <= 0 or warmup_frames < 0:
        raise ValueError("width, height and fps must be positive; warmup_frames cannot be negative")

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for node in list_video_nodes():
        candidate = inspect_node(node)
        candidate["recommended_path"] = preferred_device_path(candidate)
        if should_attempt_capture(candidate, exhaustive):
            candidate["capture"] = capture_frame(
                candidate, run_dir, width, height, fps, warmup_frames
            )
        else:
            candidate["capture"] = {
                "ok": False,
                "skipped": True,
                "error": "metadata-only node or excluded by typical-RGB-only mode",
            }
        results.append(candidate)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_profile": {"width": width, "height": height, "fps": fps},
        "exhaustive": exhaustive,
        "candidates": results,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir, results


def _print_results(run_dir: Path, results: list[dict[str, Any]]) -> None:
    print(f"\nRGB camera probe output: {run_dir}")
    if not results:
        print("No V4L2 nodes with packed RGB-compatible formats were found.")
        return
    for candidate in results:
        capture = candidate["capture"]
        status = "SAVED" if capture["ok"] else ("SKIPPED" if capture.get("skipped") else "FAILED")
        print(f"\n[{status}] {candidate['node']}")
        print(f"  formats: {', '.join(candidate['formats'])}")
        print(f"  serial: {candidate['udev'].get('ID_SERIAL_SHORT', 'unknown')}")
        print(f"  model: {candidate['udev'].get('ID_MODEL', 'unknown')}")
        print(f"  by-id: {candidate['by_id'] or '(none)'}")
        print(f"  recommended config path: {candidate['recommended_path']}")
        if capture["ok"]:
            print(f"  image: {capture['image']}")
            print(f"  channel difference: {capture['channel_difference']:.2f}")
        else:
            print(f"  error: {capture['error']}")
    print(f"\nMachine-readable report: {run_dir / 'manifest.json'}")
    print("Inspect the saved JPEG files, then assign the desired views in the recorder YAML.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find RGB-capable V4L2 nodes, resolve stable aliases, and save one frame"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rgb_camera_probe"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--typical-rgb-only",
        action="store_true",
        help="only capture nodes advertising typical packed RGB FOURCC values",
    )
    args = parser.parse_args()
    run_dir, results = probe_rgb_cameras(
        args.output_dir,
        args.width,
        args.height,
        args.fps,
        args.warmup_frames,
        exhaustive=not args.typical_rgb_only,
    )
    _print_results(run_dir, results)


if __name__ == "__main__":
    main()
