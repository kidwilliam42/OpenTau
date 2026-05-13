from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from opentau.agents.label_action_loop import LabelActionLoop, LabelActionSelector


@dataclass
class RosLabelActionLoopConfig:
    """ROS1 topic configuration for deploying the label action loop."""

    task: str
    slot: str
    image_topic: str = "/camera1/image_compressed"
    task_topic: str = "/lerobot/set_task"
    image_is_compressed: bool = True
    capture_timeout_s: float = 5.0
    cycle_period_s: float = 1.0
    camera_queue_size: int = 1
    publisher_queue_size: int = 1
    stop_command: str = "stop"


class RosImageTopicCamera:
    """Camera adapter that reads the latest image from a ROS image topic."""

    def __init__(
        self,
        topic: str,
        compressed: bool = True,
        timeout_s: float = 5.0,
        queue_size: int = 1,
        message_type: Any | None = None,
        ros_api: Any | None = None,
    ) -> None:
        self.ros_api = _load_rospy() if ros_api is None else ros_api
        self.topic = topic
        self.compressed = compressed
        self.timeout_s = timeout_s
        self._latest_image: Image.Image | None = None
        self._latest_image_monotonic_s: float | None = None
        self._lock = threading.Lock()
        self._image_ready = threading.Event()

        if message_type is None:
            message_type = _load_ros_image_message_type(compressed=compressed)

        self.subscription = self.ros_api.Subscriber(
            topic,
            message_type,
            self._on_image,
            queue_size=queue_size,
        )

    def capture(self) -> Image.Image:
        """Return the latest ROS image as a PIL RGB image."""
        if not self._image_ready.wait(timeout=self.timeout_s):
            raise TimeoutError(f"Timed out waiting for image on ROS topic {self.topic!r}.")

        with self._lock:
            if self._latest_image is None:
                raise RuntimeError(f"No image has been decoded from ROS topic {self.topic!r}.")
            if self._latest_image_monotonic_s is None:
                raise RuntimeError(f"No image timestamp has been recorded for ROS topic {self.topic!r}.")
            image_age_s = time.monotonic() - self._latest_image_monotonic_s
            if image_age_s > self.timeout_s:
                raise TimeoutError(
                    f"Latest image on ROS topic {self.topic!r} is stale "
                    f"({image_age_s:.2f}s old, timeout {self.timeout_s:.2f}s)."
                )
            return self._latest_image.copy()

    def _on_image(self, msg: Any) -> None:
        image = compressed_image_msg_to_pil(msg) if self.compressed else image_msg_to_pil(msg)
        with self._lock:
            self._latest_image = image
            self._latest_image_monotonic_s = time.monotonic()
            self._image_ready.set()


class RosInstructionExecutor:
    """VLA executor adapter that publishes instructions over one ROS String topic."""

    def __init__(
        self,
        task_topic: str = "/lerobot/set_task",
        queue_size: int = 1,
        stop_command: str = "stop",
        string_msg_type: Any | None = None,
        ros_api: Any | None = None,
    ) -> None:
        self.ros_api = _load_rospy() if ros_api is None else ros_api
        if string_msg_type is None:
            string_msg_type = _load_ros_string_message_type()

        self.task_topic = task_topic
        self.stop_command = stop_command
        self.string_msg_type = string_msg_type
        self.publisher = self.ros_api.Publisher(
            task_topic,
            string_msg_type,
            queue_size=queue_size,
        )

    def execute(self, instruction: str) -> None:
        """Publish a new VLA task instruction on /lerobot/set_task."""
        instruction = instruction.strip()
        if not instruction:
            return
        self.publisher.publish(self._make_message(instruction))

    def stop(self) -> None:
        """Publish the stop command on the same VLA task topic."""
        self.publisher.publish(self._make_message(self.stop_command))

    def _make_message(self, payload: str) -> Any:
        msg = self.string_msg_type()
        msg.data = payload
        return msg


def create_ros_label_action_loop(
    selector: LabelActionSelector,
    cfg: RosLabelActionLoopConfig,
    ros_api: Any | None = None,
) -> LabelActionLoop:
    """Create a LabelActionLoop wired to ROS topic adapters."""
    camera = RosImageTopicCamera(
        topic=cfg.image_topic,
        compressed=cfg.image_is_compressed,
        timeout_s=cfg.capture_timeout_s,
        queue_size=cfg.camera_queue_size,
        ros_api=ros_api,
    )
    executor = RosInstructionExecutor(
        task_topic=cfg.task_topic,
        queue_size=cfg.publisher_queue_size,
        stop_command=cfg.stop_command,
        ros_api=ros_api,
    )
    return LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task=cfg.task,
        slot=cfg.slot,
    )


def run_ros_label_action_loop(
    selector: LabelActionSelector,
    cfg: RosLabelActionLoopConfig,
    node_name: str = "opentau_label_action_loop",
    max_cycles: int | None = None,
) -> None:
    """Run the label action loop as a ROS1 rospy node."""
    rospy = _load_rospy()
    rospy.init_node(node_name, anonymous=False)
    loop = create_ros_label_action_loop(selector=selector, cfg=cfg, ros_api=rospy)
    rate = None if cfg.cycle_period_s <= 0 else rospy.Rate(1.0 / cfg.cycle_period_s)

    cycles = 0
    try:
        while not rospy.is_shutdown() and not loop.done:
            if max_cycles is not None and cycles >= max_cycles:
                break

            loop.step()
            cycles += 1
            if rate is not None:
                rate.sleep()
    except KeyboardInterrupt:
        _safe_stop(loop.executor, rospy)
    except Exception:
        _safe_stop(loop.executor, rospy)
        raise
    else:
        if not loop.done:
            _safe_stop(loop.executor, rospy)


def compressed_image_msg_to_pil(msg: Any) -> Image.Image:
    """Decode a sensor_msgs/CompressedImage-style message into a PIL RGB image."""
    return Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")


def image_msg_to_pil(msg: Any) -> Image.Image:
    """Decode a sensor_msgs/Image-style message into a PIL RGB image."""
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(getattr(msg, "step", 0))
    data = bytes(msg.data)

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        row_bytes = width * channels
        step = step or row_bytes
        if step < row_bytes:
            raise ValueError(f"ROS image step {step} is too small for {width}x{encoding}.")
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        array = array[:, :row_bytes].reshape(height, width, channels)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")

    if encoding in {"mono8", "8uc1"}:
        row_bytes = width
        step = step or row_bytes
        if step < row_bytes:
            raise ValueError(f"ROS image step {step} is too small for {width}x{encoding}.")
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        array = array[:, :row_bytes].reshape(height, width)
        return Image.fromarray(np.ascontiguousarray(array), mode="L").convert("RGB")

    raise ValueError(f"Unsupported ROS image encoding {encoding!r}.")


def _load_ros_image_message_type(compressed: bool) -> Any:
    try:
        from sensor_msgs.msg import CompressedImage
        from sensor_msgs.msg import Image as RosImage
    except ImportError as exc:
        raise ImportError("ROS image camera requires sensor_msgs from a ROS1 installation.") from exc

    return CompressedImage if compressed else RosImage


def _load_ros_string_message_type() -> Any:
    try:
        from std_msgs.msg import String
    except ImportError as exc:
        raise ImportError("ROS instruction executor requires std_msgs from a ROS1 installation.") from exc

    return String


def _load_rospy() -> Any:
    try:
        import rospy
    except ImportError as exc:
        raise ImportError("ROS deployment requires rospy from a ROS1 installation.") from exc

    return rospy


def _safe_stop(executor: Any, ros_api: Any) -> None:
    try:
        executor.stop()
    except Exception as exc:
        if hasattr(ros_api, "logerr"):
            ros_api.logerr("Failed to publish VLA stop command: %s", exc)
