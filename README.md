# pika_umi

基于 LeRobot 0.4.3 的 Pika 手持式数据采集与 UMI 相对轨迹训练适配项目。

项目当前聚焦两件事：

1. 从 Pika Sense、Vive Tracker 和腕部相机采集保留完整绝对量的 Raw LeRobot 数据；
2. 在训练读取阶段把绝对 TCP 轨迹转换为以当前帧为参考的相对 observation/action，供 ACT 或 Diffusion Policy 使用。

采集端是只读设备，不连接机械臂，也不会发送控制命令。在线机器人推理不属于当前版本的验收范围。

## 项目状态

当前已经实现：

- Pika Sense、Vive Tracker、夹爪宽度和一个或多个相机的 LeRobot 采集适配；
- Tracker 到 TCP 的固定外参变换；
- Raw `LeRobotDataset` 保存和 episode 划分；
- 数据集检查命令；
- ACT 的 UMI 相对轨迹 wrapper；
- Diffusion 的显式 observation 时间窗口 wrapper；
- 严格未来动作窗口 `[t+1, ..., t+Hact]`；
- episode 边界 padding 和 Diffusion action padding loss mask；
- 相对状态/动作统计量；
- wrapper、几何计算和训练入口的轻量测试。

当前尚未完成：

- 相机、Tracker、夹爪的硬件时间戳同步；
- 在安装 LeRobot 0.4.3 的 Python 3.10+ 环境中完成真实 Diffusion forward/backward smoke test；
- Pika/xArm 在线推理和动作调度；
- 严格复现原始 UMI 的 rotation-6D、latency matching 和完整部署栈。

因此当前版本适合继续进行数据采集验证和训练链路试验。正式长时间训练前，请先完成本文末尾 TODO 中的 P0 项目。

## 总体架构

```text
Pika handheld
├── Vive Tracker absolute pose
├── Tracker-to-TCP calibrated absolute TCP pose
├── Pika gripper width
├── RGB camera image(s)
└── local sensor timestamp / sample id
                    │
                    ▼
             Raw LeRobotDataset
        absolute state + absolute action
                    │
                    ▼
       training-time UMI dataset wrapper
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
ACT wrapper               Diffusion wrapper
state: (Hobs*7,)           state: (Hobs,7)
image: current sample      image: (Hobs,C,H,W)
action: (Hact,7)           action: (Hact,7)
        │                       │
        └───────────┬───────────┘
                    ▼
          LeRobot official trainer
 policy / optimizer / scheduler / checkpoint
```

### 相对位姿语义

对于当前时刻 `t`，所有历史状态和未来动作都使用当前 TCP `T_t` 作为参考：

```text
observation[i] = inverse(T_t) @ T_i
action[j]      = inverse(T_t) @ T_j
```

默认窗口：

```text
observation_horizon = 2
action_horizon      = 16

observation: [t-1, t]
action:      [t+1, t+2, ..., t+16]
```

每个模型侧状态/动作步为 7 维：

```text
[x, y, z, rotation-vector-x, rotation-vector-y, rotation-vector-z, gripper]
```

当前位置 `t` 的相对位姿前 6 维应为零。夹爪是独立标量，不参与 SE(3) 变换。

## 目录结构

```text
config/pika/
└── pika_direct_record.yaml          # 采集示例配置

docs/
└── pika_direct_fastumi.md           # 数据字段和适配细节

rules/
├── 81-vive.rules                    # Vive USB 规则
└── sensor_serial.rules              # 串口规则

src/lerobot_robot_ufactory/
├── devices/pika/
│   └── pika_device.py               # Sense/Tracker/Gripper SDK 包装
├── pika_direct/
│   ├── config.py                    # PikaDirectRobot 配置
│   ├── geometry.py                  # 位姿与旋转变换
│   ├── inspection.py                # 数据集检查
│   ├── robot.py                     # 只读 LeRobot Robot 适配器
│   └── umi_relative.py              # ACT/Diffusion 数据 wrapper
└── scripts/
    ├── uf_lerobot_record.py         # 采集入口
    └── uf_lerobot_train.py          # 官方 trainer 的薄适配入口

tests/                               # 几何、窗口、padding、factory 测试
```

## 环境要求

- Python `>=3.10`；
- LeRobot `0.4.3`；
- Pika SDK `agx-pypika`；
- 推荐 Ubuntu/Linux 采集环境；
- 已正确连接 Pika Sense、Vive Tracker 和相机。

安装：

```bash
git clone https://github.com/VoDophin/fast-umi-pika.git
cd fast-umi-pika
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

开发和测试依赖：

```bash
pip install -e '.[dev]'
```

## 配置采集设备

复制并修改 [`config/pika/pika_direct_record.yaml`](config/pika/pika_direct_record.yaml)：

```yaml
robot:
  type: uf::pika_direct
  id: pika_handheld
  port: /dev/ttyUSB0
  tracker_device_id: T20
  tracker_to_tcp:
    translation_m: [0.0, 0.0, 0.0]
    rotation_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
  gripper_closed_width_mm: 0.4
  gripper_open_width_mm: 98.1
  cameras:
    pika:
      type: opencv
      index_or_path: /dev/v4l/by-id/REPLACE_PIKA_CAMERA
      width: 640
      height: 480
      fps: 30
      fourcc: MJPG

dataset:
  root: ./data/pika_direct
  repo_id: local/pika_direct
  single_task: REPLACE_TASK_DESCRIPTION
  fps: 30
  episode_time_s: 30
  num_episodes: 50
  video: true
  push_to_hub: false
```

必须确认：

- `port` 对应目标 Pika Sense；
- `tracker_device_id` 对应这台 Pika 上的 Tracker，而不是 Lighthouse (`LH*`)；
- `tracker_to_tcp` 是实际标定值；
- 相机路径、分辨率和 FPS 可正常打开；
- 夹爪开闭宽度与实际设备一致。

## 采集 Raw 数据

```bash
uf-lerobot-record --config_path=config/pika/pika_direct_record.yaml
```

每个 episode 开始前按一次 Enter。采集过程中再次按 Enter，可以提前结束当前 episode：已有有效帧时会调用 `save_episode()`，没有有效帧时不会保存空 episode。之后程序继续等待你按 Enter 开始下一个 episode；达到配置的 `num_episodes` 后才结束整个采集并断开设备。未提前结束时，episode 达到 `episode_time_s` 后自动保存。无效 Tracker、夹爪或图像帧会被丢弃。

Raw `observation.state` 为 18 维，字段顺序固定为：

| 索引 | 字段 | 维度 |
|---|---|---:|
| `0:7` | Tracker `xyz + quaternion xyzw` | 7 |
| `7:14` | TCP `xyz + quaternion xyzw` | 7 |
| `14` | 归一化夹爪，0 关、1 开 | 1 |
| `15` | 原始夹爪宽度，毫米 | 1 |
| `16` | `time.monotonic()` 时间戳 | 1 |
| `17` | sample id | 1 |

Raw `action` 为 8 维：

```text
absolute TCP xyz+xyzw + gripper
```

它由同一次 `get_observation()` 缓存得到，不会再次读取设备。

> 注意：当前实现是顺序读取 Tracker、夹爪和相机，并把它们放入同一个 LeRobot frame；这属于逻辑配对，不是 FastUMI 论文中的统一时钟和最近邻硬件时间同步。

## 检查数据

```bash
uf-pika-inspect \
  --repo-id local/pika_direct \
  --root ./data/pika_direct \
  --episode 0
```

检查内容包括帧数、图像 shape、位姿和夹爪范围、NaN、非法四元数、非单调时间戳和异常位姿跳变。

建议在批量采集前至少：

1. 采集一个短 episode；
2. 运行检查命令；
3. 人工查看视频与 TCP 轨迹是否对应；
4. 确认 Tracker ID 和外参；
5. 再开始正式采集。

## 使用 ACT 训练

```bash
uf-lerobot-train \
  --dataset.repo_id=local/pika_direct \
  --dataset.root=./data/pika_direct \
  --policy.type=act \
  --umi.observation-horizon=2 \
  --umi.action-horizon=16 \
  --output_dir=outputs/pika_umi_act
```

ACT 当前采用兼容 LeRobot 0.4.3 的展平历史：

```text
observation.state: (B, Hobs*7)
action:            (B, Hact, 7)
```

wrapper 会设置：

```text
policy.n_obs_steps = 1
policy.chunk_size  = Hact
```

`n_obs_steps=1` 表示 ACT 接收一个模型 observation；多帧状态历史已经编码在这个 observation 的展平 state 中。

## 使用 Diffusion Policy 训练

```bash
uf-lerobot-train \
  --dataset.repo_id=local/pika_direct \
  --dataset.root=./data/pika_direct \
  --policy.type=diffusion \
  --umi.observation-horizon=2 \
  --umi.action-horizon=16 \
  --output_dir=outputs/pika_umi_diffusion
```

Diffusion 保留显式 observation 时间轴：

```text
observation.state:           (B, Hobs, 7)
observation.images.<camera>: (B, Hobs, C, H, W)
action:                      (B, Hact, 7)
action_is_pad:               (B, Hact)
```

训练入口会设置：

```text
policy.n_obs_steps                 = Hobs
policy.horizon                     = Hact
policy.drop_n_last_frames          = 0
policy.do_mask_loss_for_padding    = True
```

官方 dataset factory 创建窗口时会临时使用 `Hobs + Hact` 的采样 horizon；factory 返回后立即恢复模型 horizon 为 `Hact`。wrapper 从宽窗口中切出严格未来动作，避免把当前帧零动作放入目标。

Diffusion 的 `Hact` 还必须能被当前 1D U-Net 下采样因子整除。默认 `Hact=16` 满足当前默认网络配置。

## 参数说明

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--umi.observation-horizon` | 输入的历史/当前 observation 帧数 | 2 |
| `--umi.action-horizon` | 训练和预测的未来动作长度 | 16 |
| `policy.n_action_steps` | 推理时每次执行的动作数，不决定训练 loss 的长度 | policy 默认值并限制为 `<=Hact` |
| `--umi.disable` | 禁用 UMI wrapper，退回原生 LeRobot dataset factory | false |

训练时完整的 `Hact` 有效动作参与损失；`n_action_steps` 可以在部署阶段调整。不要同时开启其他 policy 的 `use_relative_actions`，否则动作可能被相对化两次。

## 与原生 LeRobot 的关系

项目没有修改安装目录中的 LeRobot 源码，仍使用官方的：

- policy 构造；
- optimizer 和 scheduler；
- Accelerate 和多卡训练；
- checkpoint/resume；
- W&B 和 Hugging Face Hub；
- 主训练循环。

`uf-lerobot-train` 仅在运行时替换官方 trainer 已导入的 dataset factory 符号，使 factory 返回 UMI wrapper：

| 原生 LeRobot | 本项目适配 |
|---|---|
| Raw absolute state/action | 读取样本时生成 relative state/action |
| policy delta indices 定义窗口 | 所有历史/未来位姿共用当前 TCP 参考系 |
| Raw dataset stats | 相对数据 stats |
| 默认 action 时间轴 | 严格未来 `[t+1,...,t+Hact]` |
| 默认尾部处理 | 保留边界并使用 padding mask |

## 测试

完整测试：

```bash
pytest
```

代码检查：

```bash
ruff check src tests
```

真实训练环境的最小验收应包含：

```text
LeRobotDataset
→ wrapper
→ DataLoader batch
→ policy forward
→ loss.backward()
→ optimizer.step()
→ checkpoint save/load
```

## TODO List

### P0：训练前必须完成

- [ ] 为相机、Tracker 和夹爪保存各自的采集时间戳；
- [ ] 使用独立缓冲队列，以图像时间匹配最近的 Tracker/夹爪样本；
- [ ] 记录并检查 `image_tracker_skew_ms` 和 `image_gripper_skew_ms`；
- [ ] 在 Python 3.10+、LeRobot 0.4.3 环境运行真实 ACT forward/backward smoke test；
- [ ] 在同一环境运行真实 Diffusion forward/backward smoke test；
- [ ] 用少量真实数据完成 overfit test，确认 loss 能明显下降；
- [ ] 人工验证视频、TCP 和夹爪在快速运动下仍然对齐。

### P1：训练质量

- [ ] 比较 `rotation-vector` 与原始 UMI `rotation-6D` 的训练稳定性；
- [ ] 验证相对 stats 不包含 padding 重复值；
- [ ] 增加 train/validation episode 划分和离线指标；
- [ ] 对不同 `Hobs/Hact`、FPS 和图像尺寸做消融实验；
- [ ] 增加数据质量报告和异常 episode 自动过滤；
- [ ] 评估是否导出为原始 UMI ReplayBuffer/Zarr 进行交叉验证。

### P2：以后部署

- [ ] 将相对动作块还原为绝对 TCP 目标；
- [ ] 增加推理延迟测量和 latency matching；
- [ ] 增加过期动作过滤、动作时间戳和 waypoint 调度；
- [ ] 接入目标机械臂和夹爪；
- [ ] 完成限速、工作空间、急停和碰撞安全约束。

## 已知限制

1. 当前 `sensor_timestamp` 是读取 Tracker 和夹爪后、读取相机前生成的本地单调时间，不代表相机真实曝光时间；
2. Raw 数据可以正常保存，但同步误差可能降低精细任务的模型效果；
3. 当前位姿表示是 `xyz + rotation-vector`，不是原始 UMI 的模型侧 rotation-6D；
4. ACT 的历史 state 被展平，Diffusion 才保留显式 state/image 时间轴；
5. 在线推理 adapter 已有实验实现，但没有完成目标机器人实机验收；
6. 当前工作区完成了轻量逻辑测试，仍需在目标 LeRobot 环境完成真实端到端测试。

更多实现细节见 [`docs/pika_direct_fastumi.md`](docs/pika_direct_fastumi.md)。

## License

Apache-2.0，见 [`LICENSE`](LICENSE)。
