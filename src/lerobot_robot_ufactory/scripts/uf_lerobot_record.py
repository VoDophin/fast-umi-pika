"""Focused LeRobot recorder for the read-only Pika direct Robot."""

from __future__ import annotations  # 解决类定义中使用类的情况

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.scripts.lerobot_record import create_initial_features
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


def record(cfg: RecordConfig) -> LeRobotDataset:
    robot = PikaDirectRobot(cfg.robot)
    features = create_initial_features(
        action=robot.action_features, observation=robot.observation_features
    )
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,  # 数据集名称
        cfg.dataset.fps,
        root=cfg.dataset.root,  # 实际存储位置
        robot_type=robot.name,  # 记录是什么设备产生的
        features=features,  # 一帧中包含什么字段 是什么类型 什么shape 特征名到metadata的映射
        use_videos=cfg.dataset.video, # true/false 是否保存为视频 还是图片
    )
    robot.connect()
    try:
        for episode in range(cfg.dataset.num_episodes):
            input(f"Press Enter to record episode {episode} >>> ")
            deadline = time.monotonic() + cfg.dataset.episode_time_s
            while time.monotonic() < deadline:
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
                # 注意此处的帧率控制机制还比较简单
                time.sleep(max(0.0, 1 / cfg.dataset.fps - (time.monotonic() - started)))
            dataset.save_episode()
    finally:
        robot.disconnect()
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(private=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    cfg = draccus.parse(RecordConfig, Path(args.config_path), args=[])
    record(cfg)


if __name__ == "__main__":
    main()
