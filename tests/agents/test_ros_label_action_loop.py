import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image

import opentau.agents.ros_label_action_loop as ros_loop
from opentau.agents.ros_label_action_loop import (
    RosImageTopicCamera,
    RosInstructionExecutor,
    compressed_image_msg_to_pil,
    image_msg_to_pil,
)


class FakeCompressedImage:
    def __init__(self, data):
        self.data = data


class FakeImage:
    def __init__(self, data, height, width, encoding, step=None):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding
        if step is not None:
            self.step = step


class FakeString:
    def __init__(self):
        self.data = ""


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeRospy:
    def __init__(self):
        self.subscriptions = []
        self.publishers = {}
        self.log_errors = []

    def subscriber(self, topic, message_type, callback, queue_size):
        subscription = {
            "message_type": message_type,
            "topic": topic,
            "callback": callback,
            "queue_size": queue_size,
        }
        self.subscriptions.append(subscription)
        return subscription

    def publisher(self, topic, message_type, queue_size):
        publisher = FakePublisher()
        self.publishers[topic] = {
            "message_type": message_type,
            "publisher": publisher,
            "queue_size": queue_size,
        }
        return publisher

    def init_node(self, node_name, anonymous=False):
        self.node_name = node_name
        self.anonymous = anonymous

    def rate(self, hz):
        return FakeRate(hz)

    def is_shutdown(self):
        return False

    def logerr(self, message, *args):
        self.log_errors.append(message % args if args else message)


class FakeRate:
    def __init__(self, hz):
        self.hz = hz
        self.sleep_calls = 0

    def sleep(self):
        self.sleep_calls += 1


class FakeExecutor:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class FakeLoop:
    def __init__(self, done=False, fail_on_step=False):
        self.done = done
        self.fail_on_step = fail_on_step
        self.step_calls = 0
        self.executor = FakeExecutor()

    def step(self):
        self.step_calls += 1
        if self.fail_on_step:
            raise RuntimeError("planner failed")


FakeRospy.Subscriber = FakeRospy.subscriber
FakeRospy.Publisher = FakeRospy.publisher
FakeRospy.Rate = FakeRospy.rate


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (2, 1), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_compressed_image_msg_to_pil_decodes_rgb_image():
    image = compressed_image_msg_to_pil(FakeCompressedImage(_jpeg_bytes()))

    assert image.mode == "RGB"
    assert image.size == (2, 1)


def test_image_msg_to_pil_decodes_bgr8_image():
    msg = FakeImage(
        data=bytes([30, 20, 10, 60, 50, 40]),
        height=1,
        width=2,
        encoding="bgr8",
    )

    image = image_msg_to_pil(msg)

    assert image.mode == "RGB"
    assert [image.getpixel((0, 0)), image.getpixel((1, 0))] == [(10, 20, 30), (40, 50, 60)]


def test_image_msg_to_pil_uses_step_to_ignore_row_padding():
    msg = FakeImage(
        data=bytes([30, 20, 10, 60, 50, 40, 255, 255]),
        height=1,
        width=2,
        encoding="bgr8",
        step=8,
    )

    image = image_msg_to_pil(msg)

    assert [image.getpixel((0, 0)), image.getpixel((1, 0))] == [(10, 20, 30), (40, 50, 60)]


def test_ros_image_topic_camera_returns_latest_decoded_image():
    ros_api = FakeRospy()
    camera = RosImageTopicCamera(
        topic="/camera/image_compressed",
        message_type=FakeCompressedImage,
        ros_api=ros_api,
    )

    ros_api.subscriptions[0]["callback"](FakeCompressedImage(_jpeg_bytes()))
    image = camera.capture()

    assert image.mode == "RGB"
    assert image.size == (2, 1)
    assert ros_api.subscriptions[0]["topic"] == "/camera/image_compressed"


def test_ros_image_topic_camera_rejects_stale_images():
    ros_api = FakeRospy()
    camera = RosImageTopicCamera(
        topic="/camera/image_compressed",
        timeout_s=0.1,
        message_type=FakeCompressedImage,
        ros_api=ros_api,
    )
    ros_api.subscriptions[0]["callback"](FakeCompressedImage(_jpeg_bytes()))
    camera._latest_image_monotonic_s = time.monotonic() - 1.0

    with pytest.raises(TimeoutError, match="stale"):
        camera.capture()


def test_ros_instruction_executor_publishes_plain_instruction_and_stop():
    ros_api = FakeRospy()
    executor = RosInstructionExecutor(
        task_topic="/lerobot/set_task",
        string_msg_type=FakeString,
        ros_api=ros_api,
    )

    executor.execute("Navigate to target slot A1.")
    executor.stop()

    task_pub = ros_api.publishers["/lerobot/set_task"]["publisher"]
    assert task_pub.messages[0].data == "Navigate to target slot A1."
    assert task_pub.messages[1].data == "stop"
    assert ros_api.publishers["/lerobot/set_task"]["queue_size"] == 1


def test_ros_instruction_executor_ignores_empty_instruction():
    ros_api = FakeRospy()
    executor = RosInstructionExecutor(
        task_topic="/lerobot/set_task",
        string_msg_type=FakeString,
        ros_api=ros_api,
    )

    executor.execute("  ")

    task_pub = ros_api.publishers["/lerobot/set_task"]["publisher"]
    assert task_pub.messages == []


def test_run_ros_label_action_loop_stops_when_max_cycles_reached(monkeypatch):
    ros_api = FakeRospy()
    loop = FakeLoop()
    monkeypatch.setattr(ros_loop, "_load_rospy", lambda: ros_api)
    monkeypatch.setattr(ros_loop, "create_ros_label_action_loop", lambda **kwargs: loop)

    ros_loop.run_ros_label_action_loop(
        selector=object(),
        cfg=SimpleNamespace(cycle_period_s=1.0),
        max_cycles=0,
    )

    assert loop.step_calls == 0
    assert loop.executor.stop_calls == 1


def test_run_ros_label_action_loop_stops_and_reraises_on_exception(monkeypatch):
    ros_api = FakeRospy()
    loop = FakeLoop(fail_on_step=True)
    monkeypatch.setattr(ros_loop, "_load_rospy", lambda: ros_api)
    monkeypatch.setattr(ros_loop, "create_ros_label_action_loop", lambda **kwargs: loop)

    with pytest.raises(RuntimeError, match="planner failed"):
        ros_loop.run_ros_label_action_loop(
            selector=object(),
            cfg=SimpleNamespace(cycle_period_s=1.0),
            max_cycles=1,
        )

    assert loop.executor.stop_calls == 1
