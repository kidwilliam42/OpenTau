#!/usr/bin/env python

from __future__ import annotations

import argparse

from opentau.agents import RosLabelActionLoopConfig, run_ros_label_action_loop
from opentau.planner import DEFAULT_QWEN3_VL_MODEL, Qwen3VLActionSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLM label action loop over ROS1 topics.")
    parser.add_argument("--task", required=True, help="Task text. Include the target slot in this text.")
    parser.add_argument("--slot", required=True, help="Target slot label used in VLA instruction text.")
    parser.add_argument("--model-name", default=DEFAULT_QWEN3_VL_MODEL, help="Qwen3-VL model name.")
    parser.add_argument("--device", default="cuda", help="Device for Qwen3-VL inference, e.g. cuda or cpu.")
    parser.add_argument("--image-topic", default="/camera1/image_compressed", help="ROS image topic.")
    parser.add_argument(
        "--raw-image",
        action="store_true",
        help="Use sensor_msgs/Image instead of sensor_msgs/CompressedImage.",
    )
    parser.add_argument(
        "--task-topic",
        default="/lerobot/set_task",
        help="ROS std_msgs/String topic for VLA task instructions.",
    )
    parser.add_argument(
        "--cycle-period-s",
        type=float,
        default=1.0,
        help="Seconds between VLM decision cycles.",
    )
    parser.add_argument(
        "--capture-timeout-s",
        type=float,
        default=5.0,
        help="Seconds to wait for the first/latest camera image.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional safety limit for debugging; omit for normal deployment.",
    )
    args = parser.parse_args()

    selector = Qwen3VLActionSelector(model_name=args.model_name, device=args.device)
    cfg = RosLabelActionLoopConfig(
        task=args.task,
        slot=args.slot,
        image_topic=args.image_topic,
        task_topic=args.task_topic,
        image_is_compressed=not args.raw_image,
        capture_timeout_s=args.capture_timeout_s,
        cycle_period_s=args.cycle_period_s,
    )
    run_ros_label_action_loop(selector=selector, cfg=cfg, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
