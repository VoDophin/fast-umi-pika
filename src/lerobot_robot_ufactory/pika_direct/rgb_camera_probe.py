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
    capture = cv2.VideoCapture(node, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            return {"ok": False, "error": "VideoCapture could not open the node"}

        formats = candidate["formats"]
        selected_fourcc = next(
            (fmt for fmt in (*PREFERRED_RGB_FORMATS, *OTHER_COLOR_FORMATS) if fmt in formats),
            None,
        )
        if selected_fourcc and len(selected_fourcc) == 4:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*selected_fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)

        frame = None
        for _ in range(max(1, warmup_frames + 1)):
            ok, current_frame = capture.read()
            if ok:
                frame = current_frame
        if frame is None:
            return {"ok": False, "error": "camera opened but returned no frame"}
        if frame.ndim != 3 or frame.shape[2] != 3:
            return {"ok": False, "error": f"unexpected frame shape: {tuple(frame.shape)}"}

        serial = candidate["udev"].get("ID_SERIAL_SHORT", "unknown-serial")
        filename = f"{Path(node).name}_{_safe_filename(serial)}.jpg"
        image_path = output_dir / filename
        if not cv2.imwrite(str(image_path), frame):
            return {"ok": False, "error": f"failed to write {image_path}"}

        sample = frame.astype(np.int16)
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
        return {
            "ok": True,
            "image": str(image_path),
            "shape": list(frame.shape),
            "selected_fourcc": selected_fourcc,
            "actual_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "actual_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "actual_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "channel_difference": channel_difference,
        }
    except Exception as exc:  # pragma: no cover - hardware/backend dependent
        return {"ok": False, "error": str(exc)}
    finally:
        capture.release()


def probe_rgb_cameras(
    output_root: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 5,
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
        if candidate["candidate_rank"] == 0:
            continue
        candidate["recommended_path"] = preferred_device_path(candidate)
        candidate["capture"] = capture_frame(
            candidate, run_dir, width, height, fps, warmup_frames
        )
        results.append(candidate)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_profile": {"width": width, "height": height, "fps": fps},
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
        status = "SAVED" if capture["ok"] else "FAILED"
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
    args = parser.parse_args()
    run_dir, results = probe_rgb_cameras(
        args.output_dir, args.width, args.height, args.fps, args.warmup_frames
    )
    _print_results(run_dir, results)


if __name__ == "__main__":
    main()
