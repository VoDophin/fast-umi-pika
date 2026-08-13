import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package_root = ROOT / "src/lerobot_robot_ufactory/pika_direct"
package = types.ModuleType("lerobot_robot_ufactory.pika_direct")
package.__path__ = [str(package_root)]
sys.modules[package.__name__] = package
for name in ("geometry", "umi_relative"):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", package_root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
package.UMIRelativeDataset = sys.modules[f"{package.__name__}.umi_relative"].UMIRelativeDataset
package.UMIDiffusionRelativeDataset = sys.modules[
    f"{package.__name__}.umi_relative"
].UMIDiffusionRelativeDataset
package.UMIWindowConfig = sys.modules[f"{package.__name__}.umi_relative"].UMIWindowConfig

script = ROOT / "src/lerobot_robot_ufactory/scripts/uf_lerobot_train.py"
spec = importlib.util.spec_from_file_location("uf_lerobot_train_test", script)
train_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_adapter)


def test_umi_cli_arguments_are_removed_before_lerobot_parser():
    args, remaining = train_adapter._extract_umi_args([
        "--umi.observation-horizon=3", "--umi.action-horizon", "20",
        "--policy.type=act", "--batch_size=8",
    ])
    assert args.observation_horizon == 3
    assert args.action_horizon == 20
    assert remaining == ["--policy.type=act", "--batch_size=8"]


def test_factory_synchronizes_act_chunk():
    class Policy:
        type = "act"
        chunk_size = 100
        n_action_steps = 100
        n_obs_steps = 1
    class Config:
        policy = Policy()
    class TrainModule:
        @staticmethod
        def make_dataset(cfg):
            return "raw"
    class Wrapped:
        def __init__(self, raw, config):
            self.raw, self.config = raw, config
        def __len__(self):
            return 1
    original = train_adapter.UMIRelativeDataset
    train_adapter.UMIRelativeDataset = Wrapped
    try:
        train_adapter.install_umi_dataset_factory(TrainModule, train_adapter.UMIWindowConfig(2, 3))
        result = TrainModule.make_dataset(Config())
    finally:
        train_adapter.UMIRelativeDataset = original
    assert result.raw == "raw"
    assert Config.policy.chunk_size == 3
    assert Config.policy.n_action_steps == 3


def test_factory_uses_temporary_sampling_horizon_for_diffusion():
    class Policy:
        type = "diffusion"
        horizon = 16
        n_action_steps = 8
        n_obs_steps = 2
        drop_n_last_frames = 7
        do_mask_loss_for_padding = False
        down_dims = (32,)

    class Config:
        policy = Policy()

    seen = {}

    class TrainModule:
        @staticmethod
        def make_dataset(cfg):
            seen["horizon"] = cfg.policy.horizon
            seen["n_obs_steps"] = cfg.policy.n_obs_steps
            return "raw"

    class Wrapped:
        def __init__(self, raw, config):
            self.raw, self.config = raw, config

        def __len__(self):
            return 1

    original = train_adapter.UMIDiffusionRelativeDataset
    train_adapter.UMIDiffusionRelativeDataset = Wrapped
    try:
        train_adapter.install_umi_dataset_factory(
            TrainModule, train_adapter.UMIWindowConfig(2, 4)
        )
        result = TrainModule.make_dataset(Config())
    finally:
        train_adapter.UMIDiffusionRelativeDataset = original

    assert result.raw == "raw"
    assert seen == {"horizon": 6, "n_obs_steps": 2}
    assert Config.policy.horizon == 4
    assert Config.policy.n_action_steps == 4
    assert Config.policy.do_mask_loss_for_padding is True
    assert Config.policy.drop_n_last_frames == 0


def test_factory_restores_diffusion_horizon_when_dataset_creation_fails():
    class Policy:
        type = "diffusion"
        horizon = 16
        n_action_steps = 8
        n_obs_steps = 2
        drop_n_last_frames = 7
        do_mask_loss_for_padding = False
        down_dims = (32,)

    class Config:
        policy = Policy()

    class TrainModule:
        @staticmethod
        def make_dataset(cfg):
            assert cfg.policy.horizon == 6
            raise RuntimeError("dataset failed")

    train_adapter.install_umi_dataset_factory(
        TrainModule, train_adapter.UMIWindowConfig(2, 4)
    )
    try:
        TrainModule.make_dataset(Config())
    except RuntimeError as error:
        assert str(error) == "dataset failed"
    else:
        raise AssertionError("dataset failure was not propagated")
    assert Config.policy.horizon == 4
