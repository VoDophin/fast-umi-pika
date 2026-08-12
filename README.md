# pika_umi

基于原 `lerobot_robot_ufactory` 项目结构精简出的 Pika 手持式 FastUMI
数据采集与 LeRobot 训练工程。

## Pipeline

```text
Pika handheld
  ├─ RGB / stereo cameras
  ├─ absolute Vive tracker pose
  ├─ tracker-to-TCP calibrated absolute TCP pose
  ├─ gripper opening
  └─ sensor timestamp
          ↓
Raw LeRobotDataset
          ↓
UMIRelativeEEProcessor
          ↓
images + relative EE history + gripper history
          ↓
ACT / Diffusion Policy
          ↓
[T_t^-1 T_t+1, ..., T_t^-1 T_t+K]
```

采集阶段不需要 Piper、xArm 或 CAN，也不会发送机械臂控制命令。

## 保留的原工程结构

```text
config/pika/                         # 采集配置
rules/                               # Pika/Vive udev 规则
src/lerobot_robot_ufactory/
├── devices/pika/                    # Pika Sense SDK 接入
├── pika_direct/                     # 直采、SE(3)、UMI dataset/processor
├── robots/utils.py                  # LeRobot Robot factory
└── scripts/
    ├── uf_lerobot_record.py         # 精简直采入口
    └── uf_lerobot_train.py          # LeRobot 训练薄适配入口
tests/
```

未包含 Piper/xArm 控制、GELLO、SpaceMouse、旧 UMI teleop、机械臂 eval 等
与手持直采无关的代码。

## 安装

```bash
conda activate uf_lerobot
git clone https://github.com/VoDophin/pika_umi.git
cd pika_umi
pip install -e .
```

根据采集机修改 `config/pika/pika_direct_record.yaml`：

- `port`：Pika Sense 串口；
- `tracker_device_id`：Vive Tracker ID；
- `tracker_to_tcp`：tracker 到 Pika TCP 的固定外参；
- `cameras`：一个或多个 RGB 相机；
- dataset 路径、任务、FPS 和 episode 长度。

## 数据采集

```bash
uf-lerobot-record --config_path=config/pika/pika_direct_record.yaml
```

原始 `LeRobotDataset` 保留：

```text
observation.images.*
observation.state:
  tracker xyz+xyzw
  TCP xyz+xyzw
  gripper (0 closed, 1 open)
  gripper width mm
  sensor timestamp
  sample id
action:
  absolute TCP xyz+xyzw + gripper
```

相机配置有左右两个 stream 时，两路都会保存。

## 检查数据

```bash
uf-pika-inspect \
  --repo-id local/pika_direct \
  --root ./data/pika_direct \
  --episode 0
```

检查帧数、FPS、图像 shape、位姿/夹爪范围、NaN、非法四元数、重复或
缺失帧、时间戳问题和异常位姿跳变。

## 训练

ACT：

```bash
uf-lerobot-train \
  --dataset.repo_id=local/pika_direct \
  --dataset.root=./data/pika_direct \
  --policy.type=act \
  --umi.observation-horizon=2 \
  --umi.action-horizon=16 \
  --output_dir=outputs/pika_umi_act
```

Diffusion Policy：

```bash
uf-lerobot-train \
  --dataset.repo_id=local/pika_direct \
  --dataset.root=./data/pika_direct \
  --policy.type=diffusion \
  --umi.observation-horizon=2 \
  --umi.action-horizon=16 \
  --output_dir=outputs/pika_umi_diffusion
```

不要同时开启 Pi policy 的 `use_relative_actions`，否则动作会被相对化两次。

## 与原生 LeRobot 的区别

没有修改安装目录中的 Hugging Face LeRobot 源代码。本项目仍使用：

- `RobotConfig/Robot` 和 `LeRobotDataset`；
- LeRobot policy factory；
- optimizer、scheduler、Accelerate、多卡；
- checkpoint/resume、W&B 和 Hub。

项目侧增加的行为：

| 原生 `lerobot-train` | 本项目 `uf-lerobot-train` |
|---|---|
| 使用 raw absolute state/action | 训练取样时生成 UMI relative state/action |
| policy delta timestamp 定义窗口 | 全部 past/future 使用同一个当前 TCP 为参考 |
| 使用 raw dataset stats | 重新计算 relative state/action stats |
| policy 自己决定 chunk/horizon | 自动与 UMI action horizon 对齐 |

训练薄适配器只替换官方 trainer 已导入的 dataset factory 返回值；官方训练循环
本身不变。传入 `--umi.disable` 可退化为原生训练行为。

更详细的数据字段和算法说明见
[`docs/pika_direct_fastumi.md`](docs/pika_direct_fastumi.md)。

## 测试

```bash
pip install -e '.[dev]'
pytest
```

当前测试覆盖 SE(3)、旋转 composition、全局刚体变换不变性、UMI observation/
action window、episode 边界、relative stats 和训练入口参数适配。
