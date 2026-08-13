import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _load_recorder():
    module_names = (
        "draccus", "lerobot",
        "lerobot.datasets", "lerobot.datasets.lerobot_dataset",
        "lerobot.datasets.pipeline_features", "lerobot.datasets.utils",
        "lerobot.processor", "lerobot.utils",
        "lerobot.utils.constants", "lerobot_robot_ufactory",
        "lerobot_robot_ufactory.pika_direct", "lerobot_robot_ufactory.pika_direct.robot",
        "uf_lerobot_record_test",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        draccus = types.ModuleType("draccus")
        draccus.parse = lambda *args, **kwargs: None
        sys.modules["draccus"] = draccus

        for name in module_names[1:9]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = object
        sys.modules["lerobot.datasets.utils"].build_dataset_frame = lambda *args, **kwargs: {}
        sys.modules["lerobot.datasets.utils"].combine_feature_dicts = (
            lambda *feature_dicts: {
                key: value for feature_dict in feature_dicts for key, value in feature_dict.items()
            }
        )
        sys.modules["lerobot.datasets.pipeline_features"].create_initial_features = (
            lambda **kwargs: kwargs
        )
        sys.modules["lerobot.datasets.pipeline_features"].aggregate_pipeline_dataset_features = (
            lambda pipeline, initial_features, use_videos: initial_features
        )
        sys.modules["lerobot.processor"].make_default_processors = (
            lambda: (object(), object(), object())
        )
        sys.modules["lerobot.utils.constants"].ACTION = "action"
        sys.modules["lerobot.utils.constants"].OBS_STR = "observation"

        package = types.ModuleType("lerobot_robot_ufactory")
        package.__path__ = []
        sys.modules[package.__name__] = package
        pika = types.ModuleType("lerobot_robot_ufactory.pika_direct")
        pika.PikaDirectRobot = object
        pika.PikaDirectRobotConfig = object
        sys.modules[pika.__name__] = pika
        robot_module = types.ModuleType("lerobot_robot_ufactory.pika_direct.robot")
        robot_module.InvalidPikaFrame = type("InvalidPikaFrame", (RuntimeError,), {})
        sys.modules[robot_module.__name__] = robot_module

        path = ROOT / "src/lerobot_robot_ufactory/scripts/uf_lerobot_record.py"
        spec = importlib.util.spec_from_file_location("uf_lerobot_record_test", path)
        recorder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = recorder
        spec.loader.exec_module(recorder)
        return recorder
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_enter_finishes_current_episode_and_continues_to_next(monkeypatch, tmp_path):
    recorder = _load_recorder()
    events = []

    class Robot:
        name = "pika_direct"
        action_features = {}
        observation_features = {}

        def __init__(self, config):
            self.disconnected = False

        def connect(self):
            events.append("robot_connected")

        def get_observation(self):
            return {"value": 1}

        def action_from_observation(self, observation):
            return {"value": 1}

        def disconnect(self):
            self.disconnected = True

    class Dataset:
        features = {}

        def __init__(self):
            self.frames = []
            self.saved = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

    dataset = Dataset()
    recorder.PikaDirectRobot = Robot

    def create_dataset(*args, **kwargs):
        events.append("dataset_created")
        return dataset

    recorder.LeRobotDataset = types.SimpleNamespace(create=create_dataset)
    recorder.build_dataset_frame = lambda *args, **kwargs: {"value": 1}
    # Each episode records one frame, then receives Enter on the next loop.
    presses = iter((False, True, False, True))
    monkeypatch.setattr(recorder, "_enter_pressed", lambda: next(presses))
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)

    cfg = recorder.RecordConfig(
        robot=object(),
        dataset=recorder.DatasetConfig(
            str(tmp_path / "dataset"), "repo", "task", num_episodes=2
        ),
    )
    result = recorder.record(cfg)

    assert result is dataset
    assert len(dataset.frames) == 2
    assert dataset.saved == 2
    assert events[:2] == ["robot_connected", "dataset_created"]


def test_dataset_features_are_aggregated_before_dataset_creation():
    recorder = _load_recorder()

    class Robot:
        action_features = {"tcp": object()}
        observation_features = {"camera": object()}

    calls = []
    recorder.make_default_processors = lambda: ("action_pipeline", "unused", "observation_pipeline")
    recorder.create_initial_features = lambda **kwargs: kwargs

    def aggregate(pipeline, initial_features, use_videos):
        calls.append((pipeline, initial_features, use_videos))
        prefix = "action" if "action" in initial_features else "observation"
        return {prefix: {"dtype": "video" if use_videos else "image"}}

    recorder.aggregate_pipeline_dataset_features = aggregate
    recorder.combine_feature_dicts = lambda *items: {
        key: value for item in items for key, value in item.items()
    }

    features = recorder._make_dataset_features(Robot(), use_videos=True)

    assert features == {
        "action": {"dtype": "video"},
        "observation": {"dtype": "video"},
    }
    assert calls == [
        ("action_pipeline", {"action": Robot.action_features}, True),
        ("observation_pipeline", {"observation": Robot.observation_features}, True),
    ]
    assert all("dtype" in feature for feature in features.values())


def test_device_connection_failure_does_not_create_dataset(monkeypatch, tmp_path):
    recorder = _load_recorder()
    dataset_created = False

    class Robot:
        name = "pika_direct"
        action_features = {}
        observation_features = {}

        def __init__(self, config):
            pass

        def connect(self):
            raise ConnectionError("second camera unavailable")

    def create_dataset(*args, **kwargs):
        nonlocal dataset_created
        dataset_created = True

    recorder.PikaDirectRobot = Robot
    recorder.LeRobotDataset = types.SimpleNamespace(create=create_dataset)
    dataset_root = tmp_path / "dataset"
    cfg = recorder.RecordConfig(
        robot=object(),
        dataset=recorder.DatasetConfig(str(dataset_root), "repo", "task"),
    )

    with pytest.raises(ConnectionError, match="second camera unavailable"):
        recorder.record(cfg)

    assert not dataset_created
    assert not dataset_root.exists()
