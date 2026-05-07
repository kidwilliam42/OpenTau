import pytest

from opentau.agents.label_action_loop import (
    LabelActionLoop,
    LabelDecision,
    decide_label_transition,
    to_pi05_instruction,
)


class FakeCamera:
    def __init__(self):
        self.captures = 0

    def capture(self):
        self.captures += 1
        return f"image-{self.captures}"


class FakeSelector:
    def __init__(self, labels):
        self.labels = list(labels)
        self.calls = []

    def select(self, image, current_label, task):
        self.calls.append(
            {
                "image": image,
                "current_label": current_label,
                "task": task,
            }
        )
        return self.labels.pop(0)


class FakeExecutor:
    def __init__(self):
        self.instructions = []
        self.stop_calls = 0

    def execute(self, instruction):
        self.instructions.append(instruction)

    def stop(self):
        self.stop_calls += 1


def test_to_pi05_instruction_maps_active_labels():
    assert to_pi05_instruction("A") == "Pick up the target object from the material bin."
    assert to_pi05_instruction("B") == "Navigate to the target slot."
    assert to_pi05_instruction("C") == "Place the held object into the target slot."
    assert to_pi05_instruction("D") == "Navigate to the material bin."


def test_to_pi05_instruction_rejects_done_label():
    with pytest.raises(ValueError):
        to_pi05_instruction("E")


def test_decide_label_transition_branches():
    assert decide_label_transition("E", "A") is LabelDecision.DONE
    assert decide_label_transition("B", "B") is LabelDecision.CONTINUE
    assert decide_label_transition("C", "B") is LabelDecision.SWITCH


def test_label_action_loop_stops_without_executing_done_label():
    camera = FakeCamera()
    selector = FakeSelector(["E"])
    executor = FakeExecutor()

    loop = LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task="Put the object into target slot A1.",
    )

    result = loop.run()

    assert result.done
    assert result.cycles == 1
    assert result.current_label is None
    assert executor.instructions == []
    assert executor.stop_calls == 1
    assert result.steps[0].decision is LabelDecision.DONE


def test_label_action_loop_continues_same_label_without_restarting_executor():
    camera = FakeCamera()
    selector = FakeSelector(["B", "B", "C", "E"])
    executor = FakeExecutor()

    loop = LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task="Put the object into target slot A1.",
    )

    result = loop.run()

    assert result.done
    assert result.cycles == 4
    assert result.current_label == "C"
    assert executor.instructions == [
        "Navigate to the target slot.",
        "Place the held object into the target slot.",
    ]
    assert executor.stop_calls == 1
    assert [call["current_label"] for call in selector.calls] == [None, "B", "B", "C"]
    assert [step.decision for step in result.steps] == [
        LabelDecision.SWITCH,
        LabelDecision.CONTINUE,
        LabelDecision.SWITCH,
        LabelDecision.DONE,
    ]


def test_label_action_loop_can_run_bounded_cycles_without_done():
    camera = FakeCamera()
    selector = FakeSelector(["A", "A", "A"])
    executor = FakeExecutor()
    loop = LabelActionLoop(
        camera=camera,
        selector=selector,
        executor=executor,
        task="Pick the object",
    )

    result = loop.run(max_cycles=3)

    assert not result.done
    assert result.cycles == 3
    assert result.current_label == "A"
    assert executor.instructions == ["Pick up the target object from the material bin."]
    assert executor.stop_calls == 0
