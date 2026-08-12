"""LeRobot 0.4.3 training entry with a FastUMI dataset-factory adapter."""

from __future__ import annotations

import argparse
import logging
import sys

from lerobot_robot_ufactory.pika_direct import UMIRelativeDataset, UMIWindowConfig


def _extract_umi_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--umi.observation-horizon", dest="observation_horizon", type=int, default=2)
    parser.add_argument("--umi.action-horizon", dest="action_horizon", type=int, default=16)
    parser.add_argument("--umi.disable", dest="disable", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def install_umi_dataset_factory(train_module, config: UMIWindowConfig) -> None:
    """Patch only the factory symbol imported by LeRobot's official trainer."""
    original_make_dataset = train_module.make_dataset

    def make_umi_dataset(cfg):
        policy = cfg.policy
        policy_type = policy.type
        if policy_type == "act":
            policy.chunk_size = config.action_horizon
            policy.n_action_steps = min(policy.n_action_steps, config.action_horizon)
            policy.n_obs_steps = 1  # history is already encoded in state
        elif policy_type == "diffusion":
            policy.horizon = config.action_horizon
            policy.n_action_steps = min(policy.n_action_steps, config.action_horizon)
            policy.n_obs_steps = 1  # avoid applying a second temporal history
            policy.drop_n_last_frames = 0  # wrapper already removes incomplete windows
            downsampling_factor = 2 ** len(policy.down_dims)
            if config.action_horizon % downsampling_factor:
                raise ValueError(
                    f"Diffusion UMI action horizon {config.action_horizon} must be divisible by "
                    f"the U-Net downsampling factor {downsampling_factor}"
                )
        elif getattr(policy, "use_relative_actions", False):
            raise ValueError(
                "Disable policy.use_relative_actions: FastUMI already produces relative actions"
            )
        raw = original_make_dataset(cfg)
        wrapped = UMIRelativeDataset(raw, config)
        logging.info(
            "FastUMI dataset: %d valid windows, observation_horizon=%d, action_horizon=%d",
            len(wrapped), config.observation_horizon, config.action_horizon,
        )
        return wrapped

    train_module.make_dataset = make_umi_dataset


def main() -> None:
    umi, remaining = _extract_umi_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts import lerobot_train
    if not umi.disable:
        install_umi_dataset_factory(
            lerobot_train,
            UMIWindowConfig(umi.observation_horizon, umi.action_horizon),
        )
    lerobot_train.main()


if __name__ == "__main__":
    main()
