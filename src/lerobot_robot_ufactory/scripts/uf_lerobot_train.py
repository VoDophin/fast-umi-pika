"""LeRobot 0.4.3 training entry with a FastUMI dataset-factory adapter."""

from __future__ import annotations

import argparse
import logging
import sys

from lerobot_robot_ufactory.pika_direct import (
    UMIDiffusionRelativeDataset,
    UMIRelativeDataset,
    UMIWindowConfig,
)


def _extract_umi_args(argv: list[str]):  # 提取 umi 专属参数
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--umi.observation-horizon", dest="observation_horizon", type=int, default=2)
    parser.add_argument("--umi.action-horizon", dest="action_horizon", type=int, default=16)  # 预测多少个未来
    parser.add_argument("--umi.disable", dest="disable", action="store_true")
    args, remaining = parser.parse_known_args(argv)  # umi参数被消费掉了 剩余的保留了
    return args, remaining

# 把一个运行时适配器挂载到 LeRobot 训练模块上，使其从此使用新的数据集工厂
# train_module实际上会传入lerobot_train
def install_umi_dataset_factory(train_module, config: UMIWindowConfig) -> None:
    """Patch only the factory symbol imported by LeRobot's official trainer."""
    original_make_dataset = train_module.make_dataset  # 函数对象传入

    def make_umi_dataset(cfg):
        policy = cfg.policy  # 配置里的policy信息
        policy_type = policy.type # ACT/Diffusion
        diffusion_model_horizon = None
        if policy_type == "act":
            policy.chunk_size = config.action_horizon
            policy.n_action_steps = min(policy.n_action_steps, config.action_horizon)  # 实际执行动作数
            policy.n_obs_steps = 1  # history is already encoded in state
        elif policy_type == "diffusion":
            diffusion_model_horizon = config.action_horizon
            # The official factory uses this temporary value only to sample
            # deltas [t-Hobs+1, ..., t+Hact].  The wrapper later removes the
            # past/current action prefix before the policy sees the batch.
            policy.horizon = config.observation_horizon + config.action_horizon
            policy.n_action_steps = min(policy.n_action_steps, config.action_horizon)
            policy.n_obs_steps = config.observation_horizon
            policy.drop_n_last_frames = 0
            policy.do_mask_loss_for_padding = True
            downsampling_factor = 2 ** len(policy.down_dims) # ？？
            if config.action_horizon % downsampling_factor:
                raise ValueError(
                    f"Diffusion UMI action horizon {config.action_horizon} must be divisible by "
                    f"the U-Net downsampling factor {downsampling_factor}"
                )
        elif getattr(policy, "use_relative_actions", False):
            raise ValueError(
                "Disable policy.use_relative_actions: FastUMI already produces relative actions"
            )
        if policy_type == "diffusion":
            try:
                raw = original_make_dataset(cfg)
            finally:
                # The temporary horizon exists solely to ask LeRobot for a
                # wider absolute action window.  Never leak it into the model.
                policy.horizon = diffusion_model_horizon
            wrapped = UMIDiffusionRelativeDataset(raw, config)
        else:
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
