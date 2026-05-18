#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from opentau.agents import LabelActionLoop
from opentau.planner import DEFAULT_QWEN3_VL_MODEL, Qwen3VLActionSelector


@dataclass
class DryRunEvent:
    cycle: int
    frame_index: int
    timestamp_s: float | None
    selected_label: str
    decision: str
    current_label_before: str | None
    current_label_after: str | None
    instruction: str | None
    publications: list[dict[str, str]]


class VideoFrameCamera:
    """Camera adapter that returns sampled frames from a video file."""

    def __init__(
        self,
        video_path: str | Path,
        frame_stride: int = 1,
        sample_every_s: float | None = None,
        max_image_size: int | None = 1280,
        start_frame: int = 0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Video dry-run testing requires opencv-python or opencv-python-headless.") from exc

        self.cv2 = cv2
        self.video_path = Path(video_path)
        self.frame_stride = max(1, int(frame_stride))
        self.sample_every_s = sample_every_s
        self.max_image_size = max_image_size

        self.capture_handle = cv2.VideoCapture(str(self.video_path))
        if not self.capture_handle.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.fps = float(self.capture_handle.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(self.capture_handle.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.next_frame_index = max(0, int(start_frame))
        self.last_frame_index = -1
        self.last_timestamp_s: float | None = None

        if self.next_frame_index > 0:
            self.capture_handle.set(cv2.CAP_PROP_POS_FRAMES, self.next_frame_index)

        if self.sample_every_s is not None and self.fps > 0:
            self.frame_stride = max(1, int(round(self.sample_every_s * self.fps)))

    def capture(self) -> Image.Image:
        """Return the next sampled video frame as a PIL RGB image."""
        if self.next_frame_index >= self.total_frames > 0:
            raise EOFError("Reached end of video.")

        self.capture_handle.set(self.cv2.CAP_PROP_POS_FRAMES, self.next_frame_index)
        ok, frame_bgr = self.capture_handle.read()
        if not ok:
            raise EOFError("Reached end of video.")

        self.last_frame_index = self.next_frame_index
        self.last_timestamp_s = (
            self.last_frame_index / self.fps if self.fps > 0 else None
        )
        self.next_frame_index += self.frame_stride

        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).convert("RGB")
        if self.max_image_size is not None and self.max_image_size > 0:
            image.thumbnail((self.max_image_size, self.max_image_size))
        return image

    def close(self) -> None:
        self.capture_handle.release()


class DryRunExecutor:
    """Executor that records intended ROS publications without publishing them."""

    def __init__(
        self,
        task_topic: str = "/lerobot/set_task",
        slam_command_topic: str = "/planning/perc_cmd",
        nav_bin_point: str = "A_point",
        nav_slot_point: str = "B_point",
        reset_pick_command: str = "reset_pick",
        reset_place_command: str = "reset_place",
        stop_command: str = "stop",
    ) -> None:
        self.task_topic = task_topic
        self.slam_command_topic = slam_command_topic
        self.nav_bin_point = nav_bin_point
        self.nav_slot_point = nav_slot_point
        self.reset_pick_command = reset_pick_command
        self.reset_place_command = reset_place_command
        self.stop_command = stop_command
        self.pending_publications: list[dict[str, str]] = []

    def execute(self, instruction: str) -> None:
        self.pending_publications.append(
            {"topic": self.task_topic, "type": "std_msgs/String", "data": instruction}
        )

    def execute_label(self, label: str, instruction: str) -> None:
        if label == "B":
            self.pending_publications.extend(
                [
                    {
                        "topic": self.slam_command_topic,
                        "type": "perception_msgs/PercCmd",
                        "point_name": self.nav_slot_point,
                    },
                    {
                        "topic": self.task_topic,
                        "type": "std_msgs/String",
                        "data": self.reset_place_command,
                    },
                ]
            )
            return

        if label == "D":
            self.pending_publications.extend(
                [
                    {
                        "topic": self.slam_command_topic,
                        "type": "perception_msgs/PercCmd",
                        "point_name": self.nav_bin_point,
                    },
                    {
                        "topic": self.task_topic,
                        "type": "std_msgs/String",
                        "data": self.reset_pick_command,
                    },
                ]
            )
            return

        self.execute(instruction)

    def stop(self) -> None:
        self.pending_publications.append(
            {"topic": self.task_topic, "type": "std_msgs/String", "data": self.stop_command}
        )

    def pop_publications(self) -> list[dict[str, str]]:
        publications = list(self.pending_publications)
        self.pending_publications = []
        return publications


def run_video_dry_run(args: argparse.Namespace) -> list[DryRunEvent]:
    selector = Qwen3VLActionSelector(model_name=args.model_name, device=args.device)
    camera = VideoFrameCamera(
        video_path=args.video_path,
        frame_stride=args.frame_stride,
        sample_every_s=args.sample_every_s,
        max_image_size=args.max_image_size,
        start_frame=args.start_frame,
    )
    executor = DryRunExecutor(
        task_topic=args.task_topic,
        slam_command_topic=args.slam_command_topic,
        nav_bin_point=args.nav_bin_point,
        nav_slot_point=args.nav_slot_point,
    )
    loop = LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task=args.task,
        slot=args.slot,
    )

    events: list[DryRunEvent] = []
    try:
        while not loop.done:
            if args.max_cycles is not None and len(events) >= args.max_cycles:
                break

            try:
                step = loop.step()
            except EOFError:
                break

            publications = executor.pop_publications()
            event = DryRunEvent(
                cycle=len(events) + 1,
                frame_index=camera.last_frame_index,
                timestamp_s=camera.last_timestamp_s,
                selected_label=step.selected_label,
                decision=step.decision.value,
                current_label_before=step.current_label_before,
                current_label_after=step.current_label_after,
                instruction=step.instruction,
                publications=publications,
            )
            events.append(event)
            _print_event(event)
    finally:
        camera.close()

    if args.output_jsonl is not None:
        _write_jsonl(events, args.output_jsonl)

    return events


def _print_event(event: DryRunEvent) -> None:
    ts = "unknown" if event.timestamp_s is None else f"{event.timestamp_s:.2f}s"
    print(
        f"[cycle={event.cycle} frame={event.frame_index} t={ts}] "
        f"label={event.selected_label} decision={event.decision} "
        f"instruction={event.instruction!r}"
    )
    for publication in event.publications:
        print(f"  would publish: {publication}")


def _write_jsonl(events: list[DryRunEvent], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run the VLM label agent on a video file.")
    parser.add_argument("--video-path", required=True, help="Path to the local video file.")
    parser.add_argument("--task", required=True, help="Task text. Include the target slot in this text.")
    parser.add_argument("--slot", required=True, help="Target slot label used in VLA instruction text.")
    parser.add_argument("--model-name", default=DEFAULT_QWEN3_VL_MODEL, help="Qwen3-VL model name/path.")
    parser.add_argument("--device", default="cuda", help="Device for Qwen3-VL inference, e.g. cuda or cpu.")
    parser.add_argument("--frame-stride", type=int, default=30, help="Process every Nth video frame.")
    parser.add_argument(
        "--sample-every-s",
        type=float,
        default=None,
        help="Sample one frame every N seconds. Overrides --frame-stride when video FPS is known.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1280,
        help="Resize frames so the longest edge is at most this many pixels. Use 0 to disable.",
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="Maximum VLM cycles to run.")
    parser.add_argument("--task-topic", default="/lerobot/set_task", help="Dry-run VLA topic name.")
    parser.add_argument("--slam-command-topic", default="/planning/perc_cmd", help="Dry-run SLAM topic name.")
    parser.add_argument("--nav-bin-point", default="A_point", help="SLAM point name for the material bin.")
    parser.add_argument("--nav-slot-point", default="B_point", help="SLAM point name for the slot rack.")
    parser.add_argument("--output-jsonl", default=None, help="Optional path to save dry-run events.")
    args = parser.parse_args()

    run_video_dry_run(args)


if __name__ == "__main__":
    main()
