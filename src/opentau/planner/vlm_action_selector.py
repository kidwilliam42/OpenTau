from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

ActionLabel = Literal["A", "B", "C", "D", "E"]
ActiveActionLabel = Literal["A", "B", "C", "D"]

CANDIDATES: tuple[ActionLabel, ...] = ("A", "B", "C", "D", "E")
ACTIVE_CANDIDATES: tuple[ActiveActionLabel, ...] = ("A", "B", "C", "D")


def build_select_action_prompt(
    current_label: ActiveActionLabel | str | None,
    task: str,
    slot: str | int | None = None,
) -> str:
    """Build the constrained prompt used before fixed-token logits scoring."""
    normalized_current_label = normalize_current_label(current_label)
    task_lines = [f"Task: {task}"]
    if slot is not None:
        task_lines.append(f"Target slot: {slot}")

    return (
        "Select the robot action that should be executed now.\n\n"
        "Actions:\n"
        "A = PICK: pick up the target object from the material bin.\n"
        "B = NAV_SLOT: navigate to the target slot.\n"
        "C = PLACE: place the held object into the target slot.\n"
        "D = NAV_BIN: navigate to the material bin.\n"
        "E = DONE: the task is complete.\n\n"
        f"Current action: {normalized_current_label}\n"
        f"{chr(10).join(task_lines)}\n\n"
        "Look at the image and choose the best action the robot should execute now.\n\n"
        "Rules:\n"
        "- Output A if the robot should pick up the target object from the material bin.\n"
        "- Output B if the robot should navigate to the target slot.\n"
        "- Output C if the robot should place the held object into the target slot.\n"
        "- Output D if the robot should navigate to the material bin.\n"
        "- Output E only if the object has been placed into the target slot and the whole task is complete.\n"
        "- If the current action should continue, choose the same label as Current action.\n"
        "- If uncertain, choose the same label as Current action.\n"
        "- Choose exactly one of A, B, C, D, E.\n\n"
        "Answer:"
    )


def vlm_select_action_by_logits(
    image: Any,
    current_label: ActiveActionLabel | str | None,
    task: str,
    vlm: Any,
    token_id_fn: Callable[[ActionLabel], int],
    slot: str | int | None = None,
) -> ActionLabel:
    """
    Select one of A/B/C/D/E by comparing the VLM next-token logits.

    The VLM dependency is intentionally small: it must expose
    ``forward(image=image, prompt=prompt)`` and return either logits directly,
    an object with a ``logits`` attribute, or a mapping with a ``logits`` key.
    """
    prompt = build_select_action_prompt(current_label=current_label, task=task, slot=slot)
    forward_output = vlm.forward(image=image, prompt=prompt)
    logits = _extract_next_token_logits(forward_output)

    scores = {label: _score_label(logits, token_id_fn(label)) for label in CANDIDATES}
    return max(scores, key=scores.get)


@dataclass
class VLMActionSelector:
    """Callable wrapper for the fixed-candidate VLM logits selector."""

    vlm: Any
    token_id_fn: Callable[[ActionLabel], int]

    @classmethod
    def from_tokenizer(cls, vlm: Any, tokenizer: Any) -> "VLMActionSelector":
        return cls(vlm=vlm, token_id_fn=lambda label: token_id_from_tokenizer(tokenizer, label))

    def select(
        self,
        image: Any,
        current_label: ActiveActionLabel | str | None,
        task: str,
        slot: str | int | None = None,
    ) -> ActionLabel:
        return vlm_select_action_by_logits(
            image=image,
            current_label=current_label,
            task=task,
            vlm=self.vlm,
            token_id_fn=self.token_id_fn,
            slot=slot,
        )


def normalize_current_label(current_label: ActiveActionLabel | str | None) -> ActiveActionLabel | Literal["NONE"]:
    if current_label is None:
        return "NONE"

    if current_label == "NONE":
        return "NONE"

    if current_label in ACTIVE_CANDIDATES:
        return current_label

    if current_label == "E":
        raise ValueError("current_label cannot be E because E terminates the task loop.")

    raise ValueError(f"current_label must be one of None/A/B/C/D, got {current_label!r}")


def validate_selected_label(label: str) -> ActionLabel:
    if label not in CANDIDATES:
        raise ValueError(f"selected_label must be one of {CANDIDATES}, got {label!r}")
    return label


def token_id_from_tokenizer(tokenizer: Any, label: ActionLabel) -> int:
    """Resolve a single-token candidate id from a Hugging Face style tokenizer."""
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
    else:
        encoded = tokenizer(label, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()

    if len(token_ids) != 1:
        raise ValueError(
            f"Candidate label {label!r} must map to exactly one token, got token ids {token_ids!r}."
        )

    return int(token_ids[0])


def _extract_next_token_logits(forward_output: Any) -> Any:
    if isinstance(forward_output, dict):
        if "logits" not in forward_output:
            raise ValueError("VLM forward output mapping must contain a 'logits' key.")
        logits = forward_output["logits"]
    else:
        logits = getattr(forward_output, "logits", forward_output)

    if isinstance(logits, torch.Tensor):
        if logits.ndim == 0:
            raise ValueError("Expected vector or batched logits, got a scalar tensor.")
        if logits.ndim == 3:
            logits = logits[0, -1, :]
        elif logits.ndim == 2:
            logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
        elif logits.ndim != 1:
            raise ValueError(f"Expected logits rank 1, 2, or 3, got shape {tuple(logits.shape)}.")
        return logits

    if isinstance(logits, Sequence) and logits and isinstance(logits[0], Sequence):
        return logits[-1]

    return logits


def _score_label(logits: Any, token_id: int) -> float:
    if isinstance(logits, torch.Tensor):
        return float(logits[token_id].detach().cpu().item())

    return float(logits[token_id])
