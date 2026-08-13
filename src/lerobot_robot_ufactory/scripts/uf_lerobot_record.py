"""Focused LeRobot recorder for the read-only Pika direct Robot."""

from __future__ import annotations  # 解决类定义中使用类的情况

import argparse
import os
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR

import lerobot_robot_ufactory  # noqa: F401
from lerobot_robot_ufactory.pika_direct import PikaDirectRobot, PikaDirectRobotConfig
from lerobot_robot_ufactory.pika_direct.robot import InvalidPikaFrame


@dataclass
class DatasetConfig:
    root: str
    repo_id: str
    single_task: str
    fps: int = 30
    episode_time_s: float = 30
    num_episodes: int = 1
    video: bool = True
    push_to_hub: bool = False


@dataclass  # 自动帮你生成“存数据用的类”里那些重复的样板代码。 而你只需要定义类型名和类型以及默认值
class RecordConfig:
    robot: PikaDirectRobotConfig
    dataset: DatasetConfig


def _make_dataset_features(robot: PikaDirectRobot, use_videos: bool) -> dict:
    """Convert the robot's hardware features into LeRobot dataset metadata."""
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )


def _enter_pressed() -> bool:
    """Return immediately and consume one pending Enter key press.

    POSIX terminals become readable after a complete line is entered. Windows
    terminals expose individual key presses through msvcrt. Redirected stdin
    is ignored so a closed pipe cannot terminate recording unexpectedly.
    """
    if not sys.stdin.isatty():
        return False
    if os.name == "nt":
        import msvcrt

        pressed = False
        while msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in ("\r", "\n"):
                pressed = True
        return pressed
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def record(cfg: RecordConfig) -> LeRobotDataset:
    robot = PikaDirectRobot(cfg.robot)
    features = _make_dataset_features(robot, cfg.dataset.video)
    dataset_root = Path(cfg.dataset.root).expanduser()
    if dataset_root.exists():
        raise FileExistsError(
            f"Dataset root already exists: {dataset_root}. "
            "Choose a new dataset.root or remove the old incomplete dataset explicitly."
        )
    robot.connect()
    try:
        # Create metadata only after every configured device has connected. A
        # camera startup failure must not leave an empty dataset directory.
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=features,
            use_videos=cfg.dataset.video,
        )
        for episode in range(cfg.dataset.num_episodes):
            input(f"Press Enter to record episode {episode} >>> ")
            print("Recording... Press Enter again to finish and save this episode early.")
            deadline = time.monotonic() + cfg.dataset.episode_time_s
            recorded_frames = 0
            while time.monotonic() < deadline:
                if _enter_pressed():
                    print(f"Finishing episode {episode} early.")
                    break
                started = time.monotonic()
                try:
                    observation = robot.get_observation()  ##
                except InvalidPikaFrame as exc:
                    print(f"Drop invalid Pika frame: {exc}")
                    continue
                action = robot.action_from_observation(observation)  ## 将当前的observation转化action
                dataset.add_frame(
                    {
                        **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                        **build_dataset_frame(dataset.features, action, prefix=ACTION),
                        "task": cfg.dataset.single_task,
                    }
                )  # 存入帧数据
                recorded_frames += 1
                # 注意此处的帧率控制机制还比较简单
                time.sleep(max(0.0, 1 / cfg.dataset.fps - (time.monotonic() - started)))
            if recorded_frames:
                dataset.save_episode()
                print(f"Saved episode {episode} with {recorded_frames} frames.")
            else:
                print(f"Episode {episode} contains no valid frames and was not saved.")
        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(private=False)
        return dataset
    finally:
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    cfg = draccus.parse(RecordConfig, Path(args.config_path), args=[])
    record(cfg)


if __name__ == "__main__":
    main()
