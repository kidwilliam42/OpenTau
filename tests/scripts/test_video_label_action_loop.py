from types import SimpleNamespace

from PIL import Image

from opentau.scripts.video_label_action_loop import DryRunExecutor, run_video_dry_run


class FakeSelector:
    def __init__(self, labels):
        self.labels = list(labels)
        self.calls = []

    def select(self, image, current_label, task):
        self.calls.append({"image": image, "current_label": current_label, "task": task})
        return self.labels.pop(0)


def test_dry_run_executor_routes_navigation_publications():
    executor = DryRunExecutor()

    executor.execute_label("B", "Navigate to target slot A1.")
    publications = executor.pop_publications()

    assert publications == [
        {
            "topic": "/planning/perc_cmd",
            "type": "perception_msgs/PercCmd",
            "point_name": "B_point",
        },
        {"topic": "/lerobot/set_task", "type": "std_msgs/String", "data": "reset_place"},
    ]


def test_run_video_dry_run_with_injected_components(monkeypatch):
    class FakeCamera:
        def __init__(self):
            self.frames = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
            self.last_frame_index = -1
            self.last_timestamp_s = None
            self.closed = False

        def capture(self):
            if not self.frames:
                raise EOFError
            self.last_frame_index += 1
            self.last_timestamp_s = float(self.last_frame_index)
            return self.frames.pop(0)

        def close(self):
            self.closed = True

    camera = FakeCamera()
    selector = FakeSelector(["A", "E"])

    monkeypatch.setattr(
        "opentau.scripts.video_label_action_loop.Qwen3VLActionSelector",
        lambda model_name, device: selector,
    )
    monkeypatch.setattr(
        "opentau.scripts.video_label_action_loop.VideoFrameCamera",
        lambda **kwargs: camera,
    )

    events = run_video_dry_run(
        SimpleNamespace(
            model_name="fake-model",
            device="cpu",
            video_path="fake.mp4",
            frame_stride=1,
            sample_every_s=None,
            max_image_size=1280,
            start_frame=0,
            task="Pick and place.",
            slot="A1",
            task_topic="/lerobot/set_task",
            slam_command_topic="/planning/perc_cmd",
            nav_bin_point="A_point",
            nav_slot_point="B_point",
            max_cycles=None,
            output_jsonl=None,
        )
    )

    assert [event.selected_label for event in events] == ["A", "E"]
    assert events[0].publications == [
        {"topic": "/lerobot/set_task", "type": "std_msgs/String", "data": "pick a part"}
    ]
    assert events[1].publications == [
        {"topic": "/lerobot/set_task", "type": "std_msgs/String", "data": "stop"}
    ]
    assert camera.closed
