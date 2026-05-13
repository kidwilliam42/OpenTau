import io

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


class FakeRospy:
    def __init__(self):
        self.subscriptions = []
        self.publishers = {}

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


FakeRospy.Subscriber = FakeRospy.subscriber
FakeRospy.Publisher = FakeRospy.publisher


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
