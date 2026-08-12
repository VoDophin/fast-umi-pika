# Pika Direct / FastUMI data pipeline

This path records a handheld Pika as a read-only LeRobot robot. It never
connects to Piper or CAN and never sends an actuator command.

## What this version changes

The implementation does **not** modify the installed Hugging Face LeRobot
package. It registers one project-side `Robot` type and provides a thin
training entry point which temporarily replaces the dataset factory symbol
imported by LeRobot's trainer. The native optimizer, scheduler, Accelerate,
checkpoint/resume, W&B, Hub and policy implementations are unchanged.

Outside those LeRobot integration points, this version also adds:

- a standalone SE(3) geometry module for quaternion/matrix composition;
- Pika tracker-to-TCP calibration and normalized gripper acquisition;
- a UMI relative-window dataset adapter and relative statistics;
- an episode inspection/validation command;
- example configuration, documentation, and tests;
- a small recorder-loop branch that drops invalid Pika frames and supports a
  read-only robot's action paired with the same observation sample.

Existing Pika-to-Piper teleoperation and Piper control classes are not changed.

## Record an absolute raw episode

Edit `config/pika/pika_direct_record.yaml`, especially the tracker ID, camera,
dataset path, task, and calibrated `tracker_to_tcp`, then run:

```bash
uf-lerobot-record --config_path=config/pika/pika_direct_record.yaml
```

LeRobot 0.4.3 stores the named scalar sensor features in
`observation.state`. Its metadata names have this stable order:

```text
tracker xyz+xyzw (0:7)
tcp xyz+xyzw     (7:14)
gripper          (14; 0 closed, 1 open)
gripper width mm (15)
sensor timestamp (16)
sample id        (17)
```

The normal LeRobot dataset fields (`action`, task, timestamp, frame/episode
indices and images) are also present. `action` contains the absolute TCP
xyz+xyzw plus absolute gripper state paired with the same atomic sensor read.

## Train with UMI relative trajectories

Recommended ACT entry point:

```bash
uf-lerobot-train \
  --dataset.repo_id=local/pika_direct \
  --dataset.root=./data/pika_direct \
  --policy.type=act \
  --umi.observation-horizon=2 \
  --umi.action-horizon=16 \
  --output_dir=outputs/pika_umi_act
```

For Diffusion Policy, use the same command with
`--policy.type=diffusion`. Policy `chunk_size`/`n_action_steps` must not exceed
the configured UMI action horizon. Do not enable Pi-family
`use_relative_actions`, because UMI actions are already relative.

`uf-lerobot-train` calls LeRobot 0.4.3's official trainer after replacing only
its dataset factory result. Accelerate, policy construction, normalization,
optimizer/scheduler, checkpoint/resume, W&B and Hub behavior remain native.
The differences from `lerobot-train` are:

| Behavior | `lerobot-train` | `uf-lerobot-train` |
|---|---|---|
| Source state/action | Absolute dataset features | Raw absolute data converted at sample time |
| Temporal reference | Policy delta timestamps | One current TCP frame for all past/future poses |
| Observation state | Absolute proprioception | Relative EE history + absolute gripper history |
| Action | Absolute or policy-specific relative option | UMI relative EE chunk + absolute gripper |
| Dataset statistics | Raw feature statistics | Recomputed relative state/action statistics |
| Remaining training stack | Native LeRobot | Native LeRobot, unchanged |

Pass `--umi.disable` to make this entry point behave like the native trainer.

For a custom DataLoader, the same authoritative adapter can also be used
directly:

Wrap the raw `LeRobotDataset` with `UMIRelativeDataset` before creating the
training DataLoader:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_ufactory.pika_direct import UMIRelativeDataset, UMIWindowConfig

raw = LeRobotDataset("local/pika_direct", root="./data/pika_direct")
train_dataset = UMIRelativeDataset(
    raw,
    UMIWindowConfig(observation_horizon=2, action_horizon=16),
)
```

Each sample remains a LeRobot-style dictionary containing images, task and
indices. `observation.state` becomes flattened relative TCP history followed
by gripper history; `action` becomes `(action_horizon, 7)` containing relative
xyz+rotvec and absolute gripper state. Do not additionally enable a policy's
relative-action processor, which would apply the transform twice.

The wrapper implements Python's standard Dataset protocol and can be passed
to the same `torch.utils.data.DataLoader` and policy training loop used by
LeRobot. Configure the policy input state dimension to
`observation_horizon * 7` and its action shape to `action_horizon * 7` (or keep
the two-dimensional action tensor if that policy natively consumes chunks).

Inspect a saved episode with:

```bash
uf-pika-inspect --repo-id local/pika_direct --root ./data/pika_direct --episode 0
```
