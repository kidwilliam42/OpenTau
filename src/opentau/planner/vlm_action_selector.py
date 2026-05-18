from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch

ActionLabel = Literal["A", "B", "C", "D", "E"]
ActiveActionLabel = Literal["A", "B", "C", "D"]

DEFAULT_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
CANDIDATES: tuple[ActionLabel, ...] = ("A", "B", "C", "D", "E")
ACTIVE_CANDIDATES: tuple[ActiveActionLabel, ...] = ("A", "B", "C", "D")


def build_select_action_prompt(
    current_label: ActiveActionLabel | str | None,
    task: str,
) -> str:
    """Build the constrained prompt used before fixed-token logits scoring."""
    normalized_current_label = normalize_current_label(current_label)

    return (
        "Select the robot action that should be executed now.\n\n"
        "Actions:\n"
        "A = PICK: pick up the target object from the material bin.\n"
        "B = NAV_SLOT: navigate to the target slot.\n"
        "C = PLACE: place the held object into the target slot.\n"
        "D = NAV_BIN: navigate to the material bin.\n"
        "E = DONE: the task is complete.\n\n"
        f"Current action: {normalized_current_label}\n"
        f"Task: {task}\n\n"
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
) -> ActionLabel:
    """
    Select one of A/B/C/D/E by comparing the VLM next-token logits.

    The VLM dependency is intentionally small: it must expose
    ``forward(image=image, prompt=prompt)`` and return either logits directly,
    an object with a ``logits`` attribute, or a mapping with a ``logits`` key.
    """
    prompt = build_select_action_prompt(current_label=current_label, task=task)
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
    ) -> ActionLabel:
        return vlm_select_action_by_logits(
            image=image,
            current_label=current_label,
            task=task,
            vlm=self.vlm,
            token_id_fn=self.token_id_fn,
        )


@dataclass
class Qwen3VLActionSelector:
    """Qwen3-VL-4B-Instruct selector that scores fixed action labels by logits."""

    model_name: str = DEFAULT_QWEN3_VL_MODEL
    device: str | torch.device = "cuda"
    processor: Any | None = None
    model: Any | None = None
    torch_dtype: Any = "auto"
    device_map: Any | None = "auto"
    token_ids: dict[ActionLabel, int] = field(init=False)

    def __post_init__(self) -> None:
        self.device = str(self.device) if isinstance(self.device, torch.device) else self.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        if self.processor is None or self.model is None:
            self.processor, self.model = self._load_model_and_processor()

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.token_ids = {label: token_id_from_tokenizer(tokenizer, label) for label in CANDIDATES}
        self.input_device = self._resolve_input_device()
        if hasattr(self.model, "eval"):
            self.model.eval()

    def select(
        self,
        image: Any,
        current_label: ActiveActionLabel | str | None,
        task: str,
    ) -> ActionLabel:
        return vlm_select_action_by_logits(
            image=image,
            current_label=current_label,
            task=task,
            vlm=self,
            token_id_fn=self.token_ids.__getitem__,
        )

    def forward(self, image: Any, prompt: str) -> Any:
        """Run one Qwen3-VL forward pass and return next-token logits."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        inputs = _move_inputs_to_device(inputs, self.input_device)

        with torch.inference_mode():
            return self.model(**inputs)

    def _load_model_and_processor(self) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL action selection requires transformers with AutoProcessor "
                "and AutoModelForImageTextToText support."
            ) from exc

        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model_class = self._resolve_qwen3_vl_model_class(AutoModelForImageTextToText)
        if self.device.startswith("cuda"):
            model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )
        else:
            model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            )
            model.to(torch.device(self.device))

        return processor, model

    def _resolve_qwen3_vl_model_class(self, fallback_model_class: Any) -> Any:
        """Use the dedicated Qwen3-VL class when available, otherwise fall back to AutoModel."""
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            return fallback_model_class

    def _resolve_input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device(self.device)


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


def _move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    if hasattr(inputs, "to"):
        return inputs.to(device)

    return inputs
