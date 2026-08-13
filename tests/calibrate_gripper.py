import math
import statistics
import time

from lerobot_robot_ufactory.devices.pika import PikaDevice


PIKA_SENSE_PORT = "/dev/ttyUSB0"
TRACKER_DEVICE_ID = "T20"

SAMPLE_COUNT = 50
SAMPLE_INTERVAL_S = 0.02


def sample_gripper_width(sense, sample_count: int = SAMPLE_COUNT) -> float:
    """连续读取夹爪宽度，过滤无效数据并返回中位数。"""
    values: list[float] = []

    while len(values) < sample_count:
        width = sense.get_gripper_distance()

        if width is not None:
            width = float(width)

            if math.isfinite(width):
                values.append(width)
                print(
                    f"\r采样进度：{len(values):02d}/{sample_count}，"
                    f"当前读数：{width:.3f} mm",
                    end="",
                    flush=True,
                )

        time.sleep(SAMPLE_INTERVAL_S)

    print()
    return float(statistics.median(values))


def main() -> None:
    device = PikaDevice(
        1,
        pika_sense_port=PIKA_SENSE_PORT,
        pika_tracker_device=TRACKER_DEVICE_ID,
    )
    sense = device.pika_sense

    try:
        input("\n请将夹爪完全闭合，然后按 Enter 开始采样：")
        closed_width = sample_gripper_width(sense)
        print(f"闭合标定值：{closed_width:.3f} mm")

        input("\n请将夹爪完全张开，然后按 Enter 开始采样：")
        open_width = sample_gripper_width(sense)
        print(f"张开标定值：{open_width:.3f} mm")

        if open_width <= closed_width:
            raise RuntimeError(
                "张开读数必须大于闭合读数："
                f"closed={closed_width:.3f}, "
                f"open={open_width:.3f}"
            )

        print("\n标定完成，请将下面两行写入 YAML 配置：\n")
        print(f"gripper_closed_width_mm: {closed_width:.3f}")
        print(f"gripper_open_width_mm: {open_width:.3f}")

        middle_width = (closed_width + open_width) / 2
        normalized = (middle_width - closed_width) / (
            open_width - closed_width
        )

        print("\n归一化检查：")
        print(f"闭合 {closed_width:.3f} mm → 0.000")
        print(f"中间 {middle_width:.3f} mm → {normalized:.3f}")
        print(f"张开 {open_width:.3f} mm → 1.000")

    finally:
        sense.disconnect()


if __name__ == "__main__":
    main()
