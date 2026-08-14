import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "src/lerobot_robot_ufactory/pika_direct/rgb_camera_probe.py"
SPEC = importlib.util.spec_from_file_location("rgb_camera_probe_test", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_parse_and_rank_rgb_formats():
    output = """
        [0]: 'Z16 ' (16-bit Depth)
        [1]: 'YUYV' (YUYV 4:2:2)
        [2]: 'MJPG' (Motion-JPEG)
    """
    formats = probe.parse_v4l2_formats(output)
    rank, reason = probe.rgb_candidate_rank(formats)

    assert formats == ("Z16", "YUYV", "MJPG")
    assert rank == 2
    assert "YUYV" in reason
    assert "MJPG" in reason


def test_depth_and_greyscale_only_node_is_not_rgb_candidate():
    formats = ("Z16", "GREY", "Y8I")
    rank, reason = probe.rgb_candidate_rank(formats)

    assert rank == 0
    assert "no RGB" in reason


def test_uyvy_is_retained_as_lower_confidence_candidate():
    rank, reason = probe.rgb_candidate_rank(("GREY", "UYVY"))

    assert rank == 1
    assert "possible" in reason


def test_parse_udev_properties_and_prefer_stable_alias():
    properties = probe.parse_udev_properties(
        "ID_MODEL=RealSense_D435i\nID_SERIAL_SHORT=337543060532\n"
    )
    candidate = {
        "node": "/dev/video4",
        "by_id": ["/dev/v4l/by-id/camera-337543060532"],
        "by_path": ["/dev/v4l/by-path/pci-camera"],
    }

    assert properties["ID_SERIAL_SHORT"] == "337543060532"
    assert probe.preferred_device_path(candidate) == candidate["by_id"][0]


def test_video_nodes_sort_numerically():
    nodes = [Path("/dev/video10"), Path("/dev/video2"), Path("/dev/video4")]

    assert sorted(nodes, key=probe.video_node_sort_key) == [
        Path("/dev/video2"),
        Path("/dev/video4"),
        Path("/dev/video10"),
    ]
