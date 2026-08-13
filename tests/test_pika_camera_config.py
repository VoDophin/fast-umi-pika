"""Regression checks for camera choice registration in Pika YAML configs."""

import pytest


def test_importing_pika_config_registers_opencv_camera_choice():
    pytest.importorskip("lerobot")

    import lerobot_robot_ufactory.pika_direct.config  # noqa: F401
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    # Importing this module must register all camera choices accepted by the
    # example PikaDirectRobotConfig YAML before Draccus starts decoding it.
    assert CameraConfig.get_choice_class("opencv") is OpenCVCameraConfig
