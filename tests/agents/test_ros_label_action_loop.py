import io
import json

from PIL import Image

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
    def __init__(self, data, height, width, encoding):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding


class FakeString:
    def __init__(self):
        self.data = ""


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.publishers = {}

    def create_subscription(self, message_type, topic, callback, qos_profile):
        subscription = {
            "message_type": message_type,
            "topic": topic,
            "callback": callback,
            "qos_profile": qos_profile,
        }
        self.subscriptions.append(subscription)
        return subscription

    def create_publisher(self, message_type, topic, qos_profile):
        publisher = FakePublisher()
        self.publishers[topic] = {
            "message_type": message_type,
            "publisher": publisher,
            "qos_profile": qos_profile,
        }
        return publisher


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


def test_ros_image_topic_camera_returns_latest_decoded_image():
    node = FakeNode()
    camera = RosImageTopicCamera(
        node=node,
        topic="/camera/image_compressed",
        message_type=FakeCompressedImage,
    )

    node.subscriptions[0]["callback"](FakeCompressedImage(_jpeg_bytes()))
    image = camera.capture()

    assert image.mode == "RGB"
    assert image.size == (2, 1)
    assert node.subscriptions[0]["topic"] == "/camera/image_compressed"


def test_ros_instruction_executor_publishes_plain_instruction_and_stop():
    node = FakeNode()
    executor = RosInstructionExecutor(
        node=node,
        instruction_topic="/robot/pi05_instruction",
        stop_topic="/robot/pi05_stop",
        string_msg_type=FakeString,
    )

    executor.execute("Navigate to the target slot.")
    executor.stop()

    instruction_pub = node.publishers["/robot/pi05_instruction"]["publisher"]
    stop_pub = node.publishers["/robot/pi05_stop"]["publisher"]
    assert instruction_pub.messages[0].data == "Navigate to the target slot."
    assert stop_pub.messages[0].data == "STOP"


def test_ros_instruction_executor_can_publish_json_payloads():
    node = FakeNode()
    executor = RosInstructionExecutor(
        node=node,
        instruction_topic="/robot/pi05_instruction",
        stop_topic="/robot/pi05_stop",
        publish_json=True,
        string_msg_type=FakeString,
    )

    executor.execute("Place the held object into the target slot.")
    executor.stop()

    instruction_pub = node.publishers["/robot/pi05_instruction"]["publisher"]
    stop_pub = node.publishers["/robot/pi05_stop"]["publisher"]
    assert json.loads(instruction_pub.messages[0].data) == {
        "command": "execute",
        "instruction": "Place the held object into the target slot.",
    }
    assert json.loads(stop_pub.messages[0].data) == {"command": "stop", "instruction": "STOP"}
