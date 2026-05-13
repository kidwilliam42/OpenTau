from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from opentau.planner.vlm_action_selector import (
    ACTIVE_CANDIDATES,
    ActionLabel,
    ActiveActionLabel,
    validate_selected_label,
)


class LabelDecision(str, Enum):
    """Decision branches after the VLM selects a label."""

    DONE = "done"
    CONTINUE = "continue"
    SWITCH = "switch"


class Camera(Protocol):
    """Minimal camera interface used by the agent-in-loop controller."""

    def capture(self) -> Any:
        """Capture the latest image for VLM action selection."""
        ...


class LabelActionSelector(Protocol):
    """Minimal VLM action selector interface."""

    def select(
        self,
        image: Any,
        current_label: ActiveActionLabel | None,
        task: str,
    ) -> ActionLabel:
        """Return the selected action label from fixed candidates A/B/C/D/E."""
        ...


class Pi05Executor(Protocol):
    """Instruction executor for a low-level pi0.5 policy or service."""

    def execute(self, instruction: str) -> None:
        """Start or switch the low-level executor to the given instruction."""
        ...

    def stop(self) -> None:
        """Stop the executor when the task is complete."""
        ...


@dataclass
class LabelLoopStep:
    """Debug record for one VLM decision cycle."""

    image: Any
    selected_label: ActionLabel
    decision: LabelDecision
    current_label_before: ActiveActionLabel | None
    current_label_after: ActiveActionLabel | None
    instruction: str | None = None


@dataclass
class LabelLoopResult:
    """Summary returned by a completed or bounded label loop run."""

    done: bool
    cycles: int
    current_label: ActiveActionLabel | None
    steps: list[LabelLoopStep] = field(default_factory=list)


def decide_label_transition(
    selected_label: ActionLabel | str,
    current_label: ActiveActionLabel | None,
) -> LabelDecision:
    """Implement the E / same label / different label decision logic."""
    selected_label = validate_selected_label(selected_label)

    if selected_label == "E":
        return LabelDecision.DONE

    if selected_label == current_label:
        return LabelDecision.CONTINUE

    return LabelDecision.SWITCH


def to_pi05_instruction(label: ActiveActionLabel | str, slot: str | int | None = None) -> str:
    """Map an active label to the pi0.5 natural-language instruction."""
    if label == "A":
        return "pick a part"

    if label == "B":
        if slot is None:
            raise ValueError("slot is required for label B")
        return f"Navigate to target slot {slot}."

    if label == "C":
        if slot is None:
            raise ValueError("slot is required for label C")
        return f"place a part into the box with {slot} label."

    if label == "D":
        return "Navigate to the material bin."

    raise ValueError(label)


class LabelActionLoop:
    """
    First-version robot task loop controlled by VLM label logits.

    Each cycle captures one image, asks the selector for one label from
    A/B/C/D/E, then applies the required branch:
      - E: stop executor and finish,
      - same label: keep current executor running,
      - different active label: send a new pi0.5 instruction.
    """

    def __init__(
        self,
        camera: Camera,
        selector: LabelActionSelector,
        executor: Pi05Executor,
        task: str,
        slot: str | int | None = None,
    ) -> None:
        self.camera = camera
        self.selector = selector
        self.executor = executor
        self.task = task
        self.slot = slot

        self.current_label: ActiveActionLabel | None = None
        self.done = False
        self.steps: list[LabelLoopStep] = []

    def reset(self) -> None:
        self.current_label = None
        self.done = False
        self.steps = []

    def step(self) -> LabelLoopStep:
        """Run one capture/select/decision cycle."""
        if self.done:
            raise RuntimeError("LabelActionLoop is already done.")

        image = self.camera.capture()
        current_label_before = self.current_label
        selected_label = self.selector.select(
            image=image,
            current_label=current_label_before,
            task=self.task,
        )
        selected_label = validate_selected_label(selected_label)
        decision = decide_label_transition(
            selected_label=selected_label,
            current_label=current_label_before,
        )

        instruction = None
        if decision is LabelDecision.DONE:
            self.executor.stop()
            self.done = True
        elif decision is LabelDecision.SWITCH:
            if selected_label not in ACTIVE_CANDIDATES:
                raise ValueError(f"Cannot switch to inactive label {selected_label!r}")

            instruction = to_pi05_instruction(selected_label, self.slot)
            self.executor.execute(instruction)
            self.current_label = selected_label

        step = LabelLoopStep(
            image=image,
            selected_label=selected_label,
            decision=decision,
            current_label_before=current_label_before,
            current_label_after=self.current_label,
            instruction=instruction,
        )
        self.steps.append(step)
        return step

    def run(self, max_cycles: int | None = None) -> LabelLoopResult:
        """
        Run until DONE or until ``max_cycles`` is reached.

        ``max_cycles`` is optional for production loops and useful for tests or
        simulations where the selector may intentionally never return E.
        """
        cycles = 0
        while not self.done:
            if max_cycles is not None and cycles >= max_cycles:
                break

            self.step()
            cycles += 1

        return LabelLoopResult(
            done=self.done,
            cycles=cycles,
            current_label=self.current_label,
            steps=list(self.steps),
        )
