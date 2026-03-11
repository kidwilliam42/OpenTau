from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


@dataclass
class SubtaskPlan:
    """One planner-produced subtask."""

    instruction: str
    max_steps: int


@dataclass
class ExecutionRecord:
    """Execution summary for one completed subtask."""

    instruction: str
    executed_steps: int


class NextSubtaskPlanner(Protocol):
    """Minimal interface required by HierarchicalAgent."""

    def plan_next(
        self,
        task: str,
        image_dict: dict[str, torch.Tensor],
        history: list[ExecutionRecord],
    ) -> SubtaskPlan | None:
        """
        Args:
            task: Episode-level task text.
            image_dict: Current planner-visible image tensors with batch dimension.
            history: Previously executed subtasks.

        Returns:
            The next subtask, or None if the task is complete.
        """
        ...


class HierarchicalAgent:
    """
    Minimal online hierarchical agent.

    The agent performs rolling single-subtask planning:
      1. plan one subtask from the latest observation,
      2. execute that subtask for a fixed step budget,
      3. replan from the newest observation once the budget is exhausted.

    This implementation intentionally stays simple:
      - batch_size must be 1
      - no value model
      - no explicit failure detection beyond environment success
    """

    def __init__(
        self,
        planner: NextSubtaskPlanner,
        low_level_policy: Any,
        task: str,
        default_subtask_steps: int = 15,
        max_subtasks: int = 20,
        image_keys: list[str] | None = None,
    ) -> None:
        self.planner = planner
        self.low_level_policy = low_level_policy
        self.task = task
        self.default_subtask_steps = default_subtask_steps
        self.max_subtasks = max_subtasks
        self.image_keys = image_keys

        self.reset()

    def reset(self) -> None:
        """Reset the agent at the start of a new episode."""
        self.low_level_policy.reset()

        self.history: list[ExecutionRecord] = []
        self.current_subtask: SubtaskPlan | None = None
        self.current_subtask_steps = 0
        self.needs_planning = True
        self.episode_done = False
        self.num_plans_made = 0

    def is_finished(self) -> bool:
        """Return whether the episode should stop."""
        return self.episode_done

    def current_instruction(self) -> str | None:
        """Return the active subtask instruction for logging/debugging."""
        if self.current_subtask is None:
            return None
        return self.current_subtask.instruction

    def act(self, observation: dict[str, Any]) -> torch.Tensor:
        """Produce one low-level action from the current observation."""
        if self.episode_done:
            raise StopIteration("HierarchicalAgent is already finished.")

        self._validate_batch_size(observation)

        if self.needs_planning:
            self._plan_next_subtask(observation)

        if self.current_subtask is None:
            raise StopIteration("No active subtask available for action selection.")

        policy_observation = self._build_policy_observation(observation)
        return self.low_level_policy.select_action(policy_observation)

    def update_after_env_step(self, info: dict[str, Any]) -> None:
        """
        Update agent state after env.step(...).

        Replanning is triggered when the current subtask budget is exhausted.
        """
        if self.episode_done:
            return

        if self._success_from_info(info):
            self.episode_done = True
            return

        if self.current_subtask is None:
            return

        self.current_subtask_steps += 1

        if self.current_subtask_steps >= self.current_subtask.max_steps:
            self.history.append(
                ExecutionRecord(
                    instruction=self.current_subtask.instruction,
                    executed_steps=self.current_subtask_steps,
                )
            )
            self.current_subtask = None
            self.current_subtask_steps = 0
            self.needs_planning = True

            # Clear any cached action chunk that was generated for the old prompt.
            self.low_level_policy.reset()

            if len(self.history) >= self.max_subtasks:
                self.episode_done = True

    def debug_state(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable debug snapshot."""
        return {
            "task": self.task,
            "episode_done": self.episode_done,
            "needs_planning": self.needs_planning,
            "num_plans_made": self.num_plans_made,
            "current_subtask": None
            if self.current_subtask is None
            else {
                "instruction": self.current_subtask.instruction,
                "max_steps": self.current_subtask.max_steps,
                "current_step": self.current_subtask_steps,
            },
            "history": [
                {"instruction": item.instruction, "executed_steps": item.executed_steps}
                for item in self.history
            ],
        }

    def _plan_next_subtask(self, observation: dict[str, Any]) -> None:
        """Plan the next subtask from the latest observation."""
        if self.num_plans_made >= self.max_subtasks:
            self.current_subtask = None
            self.current_subtask_steps = 0
            self.needs_planning = False
            self.episode_done = True
            return

        image_dict = self._extract_image_dict(observation)
        planned = self.planner.plan_next(
            task=self.task,
            image_dict=image_dict,
            history=self.history,
        )
        self.num_plans_made += 1

        if planned is None:
            # Some VLMs occasionally return "done" on the very first planning step
            # before any subtask has been executed. Fall back to the original task
            # so the rollout can proceed instead of terminating immediately.
            if not self.history:
                self.current_subtask = SubtaskPlan(
                    instruction=self.task,
                    max_steps=self.default_subtask_steps,
                )
                self.current_subtask_steps = 0
                self.needs_planning = False
                self.low_level_policy.reset()
                return

            self.current_subtask = None
            self.current_subtask_steps = 0
            self.needs_planning = False
            self.episode_done = True
            return

        instruction = planned.instruction.strip() or self.task
        max_steps = int(planned.max_steps)
        if max_steps <= 0:
            max_steps = self.default_subtask_steps

        self.current_subtask = SubtaskPlan(instruction=instruction, max_steps=max_steps)
        self.current_subtask_steps = 0
        self.needs_planning = False

        # New prompt means we must flush the previous low-level action queue.
        self.low_level_policy.reset()

    def _build_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Inject the active subtask as the low-level prompt.

        This matches the prompt-based interface used by pi0/pi05.
        """
        obs = deepcopy(observation)
        instruction = self.current_subtask.instruction if self.current_subtask is not None else self.task
        obs["prompt"] = [instruction]
        return obs

    def _extract_image_dict(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Collect planner-visible images from an OpenTau observation dict.

        By default, auto-detect 4D channel-first tensors shaped like [B, C, H, W].
        """
        if self.image_keys is not None:
            image_dict = {
                key: observation[key]
                for key in self.image_keys
                if key in observation and isinstance(observation[key], torch.Tensor)
            }
        else:
            image_dict = {}
            excluded = {"state", "prompt", "img_is_pad", "action_is_pad", "actions"}
            for key, value in observation.items():
                if key in excluded:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[1] in (1, 3):
                    image_dict[key] = value

        if not image_dict:
            raise ValueError(
                "No image tensors found in observation. "
                "Pass explicit image_keys if auto-detection does not match your observation format."
            )

        return image_dict

    def _validate_batch_size(self, observation: dict[str, Any]) -> None:
        """The minimal implementation supports batch_size=1 only."""
        batch_size = None
        for value in observation.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                batch_size = value.shape[0]
                break

        if batch_size is None:
            raise ValueError("Could not infer batch size from observation.")

        if batch_size != 1:
            raise NotImplementedError(
                f"HierarchicalAgent currently supports batch_size=1 only, got {batch_size}."
            )

    def _success_from_info(self, info: dict[str, Any]) -> bool:
        """Parse success flags from vectorized env info."""
        if "is_success" not in info:
            return False

        flag = info["is_success"]

        if isinstance(flag, (bool, np.bool_)):
            return bool(flag)
        if isinstance(flag, torch.Tensor):
            return bool(flag.any().item())
        if isinstance(flag, np.ndarray):
            return bool(flag.any())
        if isinstance(flag, (list, tuple)):
            return any(bool(x) for x in flag)
        return bool(flag)
