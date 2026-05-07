from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from opentau.agents.label_action_loop import LabelActionLoop, LabelActionSelector


@dataclass
class RosLabelActionLoopConfig:
    """ROS topic configuration for deploying the label action loop."""

    task: str
    image_topic: str = "/camera1/image_compressed"
    instruction_topic: str = "/opentau/pi05_instruction"
    stop_topic: str = "/opentau/pi05_stop"
    image_is_compressed: bool = True
    capture_timeout_s: float = 5.0
    cycle_period_s: float = 0.5
    publish_json: bool = False
    stop_command: str = "STOP"


class RosImageTopicCamera:
    """Camera adapter that reads the latest image from a ROS image topic."""

    def __init__(
        self,
        node: Any,
        topic: str,
        compressed: bool = True,
        timeout_s: float = 5.0,
        qos_profile: int = 1,
        message_type: Any | None = None,
    ) -> None:
        self.node = node
        self.topic = topic
        self.compressed = compressed
        self.timeout_s = timeout_s
        self._latest_image: Image.Image | None = None
        self._lock = threading.Lock()
        self._image_ready = threading.Event()

        if message_type is None:
            message_type = _load_ros_image_message_type(compressed=compressed)

        self.subscription = node.create_subscription(
            message_type,
            topic,
            self._on_image,
            qos_profile,
        )

    def capture(self) -> Image.Image:
        """Return the latest ROS image as a PIL RGB image."""
        if not self._image_ready.wait(timeout=self.timeout_s):
            raise TimeoutError(f"Timed out waiting for image on ROS topic {self.topic!r}.")

        with self._lock:
            if self._latest_image is None:
                raise RuntimeError(f"No image has been decoded from ROS topic {self.topic!r}.")
            return self._latest_image.copy()

    def _on_image(self, msg: Any) -> None:
        image = compressed_image_msg_to_pil(msg) if self.compressed else image_msg_to_pil(msg)
        with self._lock:
            self._latest_image = image
            self._image_ready.set()


class RosInstructionExecutor:
    """pi0.5 executor adapter that publishes instructions over ROS string topics."""

    def __init__(
        self,
        node: Any,
        instruction_topic: str = "/opentau/pi05_instruction",
        stop_topic: str = "/opentau/pi05_stop",
        qos_profile: int = 10,
        publish_json: bool = False,
        stop_command: str = "STOP",
        string_msg_type: Any | None = None,
    ) -> None:
        if string_msg_type is None:
            string_msg_type = _load_ros_string_message_type()

        self.node = node
        self.instruction_topic = instruction_topic
        self.stop_topic = stop_topic
        self.publish_json = publish_json
        self.stop_command = stop_command
        self.string_msg_type = string_msg_type
        self.instruction_publisher = node.create_publisher(string_msg_type, instruction_topic, qos_profile)
        self.stop_publisher = node.create_publisher(string_msg_type, stop_topic, qos_profile)

    def execute(self, instruction: str) -> None:
        """Publish a new pi0.5 instruction for the terminal server."""
        payload = (
            json.dumps({"command": "execute", "instruction": instruction})
            if self.publish_json
            else instruction
        )
        self.instruction_publisher.publish(self._make_message(payload))

    def stop(self) -> None:
        """Publish a stop command for the terminal server."""
        payload = (
            json.dumps({"command": "stop", "instruction": self.stop_command})
            if self.publish_json
            else self.stop_command
        )
        self.stop_publisher.publish(self._make_message(payload))

    def _make_message(self, payload: str) -> Any:
        msg = self.string_msg_type()
        msg.data = payload
        return msg


def create_ros_label_action_loop(
    node: Any,
    selector: LabelActionSelector,
    cfg: RosLabelActionLoopConfig,
) -> LabelActionLoop:
    """Create a LabelActionLoop wired to ROS topic adapters."""
    camera = RosImageTopicCamera(
        node=node,
        topic=cfg.image_topic,
        compressed=cfg.image_is_compressed,
        timeout_s=cfg.capture_timeout_s,
    )
    executor = RosInstructionExecutor(
        node=node,
        instruction_topic=cfg.instruction_topic,
        stop_topic=cfg.stop_topic,
        publish_json=cfg.publish_json,
        stop_command=cfg.stop_command,
    )
    return LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task=cfg.task,
    )


def run_ros_label_action_loop(
    selector: LabelActionSelector,
    cfg: RosLabelActionLoopConfig,
    node_name: str = "opentau_label_action_loop",
    max_cycles: int | None = None,
) -> None:
    """Run the label action loop as a ROS2 node."""
    try:
        import rclpy
    except ImportError as exc:
        raise ImportError("ROS deployment requires rclpy from a ROS2 installation.") from exc

    rclpy.init()
    node = rclpy.create_node(node_name)
    loop = create_ros_label_action_loop(node=node, selector=selector, cfg=cfg)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cycles = 0
    try:
        while not loop.done:
            if max_cycles is not None and cycles >= max_cycles:
                break

            loop.step()
            cycles += 1
            if cfg.cycle_period_s > 0:
                time.sleep(cfg.cycle_period_s)
    except KeyboardInterrupt:
        loop.executor.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


def compressed_image_msg_to_pil(msg: Any) -> Image.Image:
    """Decode a sensor_msgs/CompressedImage-style message into a PIL RGB image."""
    return Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")


def image_msg_to_pil(msg: Any) -> Image.Image:
    """Decode a sensor_msgs/Image-style message into a PIL RGB image."""
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    height = int(msg.height)
    width = int(msg.width)
    data = bytes(msg.data)

    if encoding in {"rgb8", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        return Image.fromarray(array, mode="RGB")

    if encoding in {"mono8", "8uc1"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return Image.fromarray(array, mode="L").convert("RGB")

    raise ValueError(f"Unsupported ROS image encoding {encoding!r}.")


def _load_ros_image_message_type(compressed: bool) -> Any:
    try:
        from sensor_msgs.msg import CompressedImage
        from sensor_msgs.msg import Image as RosImage
    except ImportError as exc:
        raise ImportError("ROS image camera requires sensor_msgs from a ROS2 installation.") from exc

    return CompressedImage if compressed else RosImage


def _load_ros_string_message_type() -> Any:
    try:
        from std_msgs.msg import String
    except ImportError as exc:
        raise ImportError("ROS instruction executor requires std_msgs from a ROS2 installation.") from exc

    return String
