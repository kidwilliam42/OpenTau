from __future__ import annotations

import json
import re
from typing import Any

import torch
from PIL import Image

from opentau.agents.hierarchical_agent import ExecutionRecord, SubtaskPlan
from opentau.planner.utils.utils import load_prompt_library


class QwenHighLevelPlanner:
    """
    Minimal Qwen3-VL planner that emits exactly one next subtask per call.

    Expected model output schema:
        {
          "done": false,
          "next_subtask": {
            "instruction": "approach the yellow block",
            "max_steps": 15
          }
        }

    Or:
        {
          "done": true
        }
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str | torch.device = "cuda",
        max_new_tokens: int = 128,
        default_subtask_steps: int = 15,
        min_subtask_steps: int = 5,
        max_subtask_steps: int = 30,
        max_history_items: int = 10,
        prompt_library_path: str = "src/opentau/planner/qwen_prompts.yaml",
        system_prompt_key: str = "qwen_online_planner_system",
        user_prompt_key: str = "qwen_online_planner_user",
    ) -> None:
        self.model_name = model_name
        self.device = str(device) if isinstance(device, torch.device) else device
        self.max_new_tokens = max_new_tokens
        self.default_subtask_steps = default_subtask_steps
        self.min_subtask_steps = min_subtask_steps
        self.max_subtask_steps = max_subtask_steps
        self.max_history_items = max_history_items
        self.prompt_library_path = prompt_library_path
        self.system_prompt_key = system_prompt_key
        self.user_prompt_key = user_prompt_key

        self.processor, self.model = self._load_model_and_processor(model_name, self.device)
        self.input_device = torch.device(self.device)
        self.prompts_dict = self._load_prompts()

    def plan_next(
        self,
        task: str,
        image_dict: dict[str, torch.Tensor],
        history: list[ExecutionRecord],
    ) -> SubtaskPlan | None:
        """
        Plan the next subtask from the current scene.

        Returns:
            - SubtaskPlan(...) if a next subtask is available
            - None if the planner believes the task is complete
        """
        pil_images = self._image_dict_to_pil_list(image_dict)
        prompt_text = self._build_user_prompt(task=task, history=history)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_prompt()}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in pil_images]
                    + [{"type": "text", "text": prompt_text}]
                ),
            },
        ]

        raw_text = self._generate(messages=messages, pil_images=pil_images)
        parsed = self._parse_response(raw_text)

        if parsed.get("done", False):
            return None

        next_subtask = parsed.get("next_subtask", {})
        instruction = str(next_subtask.get("instruction", "")).strip() or task
        max_steps = self._sanitize_max_steps(next_subtask.get("max_steps"))
        return SubtaskPlan(instruction=instruction, max_steps=max_steps)

    def debug_plan_raw(
        self,
        task: str,
        image_dict: dict[str, torch.Tensor],
        history: list[ExecutionRecord],
    ) -> str:
        """Return the raw generation output before JSON parsing."""
        pil_images = self._image_dict_to_pil_list(image_dict)
        prompt_text = self._build_user_prompt(task=task, history=history)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_prompt()}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in pil_images]
                    + [{"type": "text", "text": prompt_text}]
                ),
            },
        ]
        return self._generate(messages=messages, pil_images=pil_images)

    def _load_model_and_processor(self, model_name: str, device: str):
        """
        Lazy-load Qwen3-VL so the module remains importable even if transformers
        in the environment is older than the required version.
        """
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL requires a transformers version that provides "
                "`Qwen3VLForConditionalGeneration`. "
                "OpenTau currently pins an older transformers version, so you will likely "
                "need to upgrade transformers or isolate this planner in a separate environment."
            ) from exc

        if device.startswith("cuda"):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
            model.to(torch.device(device))

        processor = AutoProcessor.from_pretrained(model_name)
        model.eval()
        return processor, model

    def _load_prompts(self) -> dict[str, Any]:
        """Load planner prompt templates from a YAML file."""
        prompts_dict = load_prompt_library(self.prompt_library_path)
        if prompts_dict is None:
            raise FileNotFoundError(
                f"Failed to load Qwen prompt library from '{self.prompt_library_path}'."
            )
        prompts = prompts_dict.get("prompts", {})
        if self.system_prompt_key not in prompts:
            raise KeyError(
                f"System prompt key '{self.system_prompt_key}' not found in '{self.prompt_library_path}'."
            )
        if self.user_prompt_key not in prompts:
            raise KeyError(
                f"User prompt key '{self.user_prompt_key}' not found in '{self.prompt_library_path}'."
            )
        return prompts_dict

    def _system_prompt(self) -> str:
        return self._render_template(
            self.prompts_dict["prompts"][self.system_prompt_key]["template"],
            min_subtask_steps=self.min_subtask_steps,
            max_subtask_steps=self.max_subtask_steps,
        )

    def _build_user_prompt(self, task: str, history: list[ExecutionRecord]) -> str:
        history_text = self._format_history(history)
        return self._render_template(
            self.prompts_dict["prompts"][self.user_prompt_key]["template"],
            task=task,
            executed_subtasks=history_text,
        )

    def _render_template(self, template: str, **kwargs: Any) -> str:
        """
        Render known placeholders without interpreting literal JSON braces.

        Prompt templates include JSON examples such as {"done": true}. Using
        str.format() would treat those braces as placeholders, so we only
        replace the explicit variables we own.
        """
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    def _format_history(self, history: list[ExecutionRecord]) -> str:
        if not history:
            return "None."

        trimmed = history[-self.max_history_items :]
        lines = []
        for idx, item in enumerate(trimmed, start=1):
            lines.append(f"{idx}. {item.instruction} ({item.executed_steps} steps)")
        return "\n".join(lines)

    def _generate(self, messages: list[dict[str, Any]], pil_images: list[Image.Image]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.input_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_ids = inputs.get("input_ids")
        generated_only = generated_ids[:, input_ids.shape[1] :] if input_ids is not None else generated_ids
        output_text = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return output_text.strip()

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse planner output with a few robust fallbacks."""
        text = text.strip()

        try:
            parsed = json.loads(text)
            return self._normalize_parsed_output(parsed)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_first_json_object(text)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
                return self._normalize_parsed_output(parsed)
            except json.JSONDecodeError:
                pass

        return {
            "done": False,
            "next_subtask": {
                "instruction": self._fallback_instruction_from_text(text),
                "max_steps": self.default_subtask_steps,
            },
        }

    def _normalize_parsed_output(self, parsed: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {
                "done": False,
                "next_subtask": {
                    "instruction": str(parsed),
                    "max_steps": self.default_subtask_steps,
                },
            }

        done = bool(parsed.get("done", False))
        if done:
            return {"done": True}

        next_subtask = parsed.get("next_subtask", {})
        if not isinstance(next_subtask, dict):
            next_subtask = {}

        instruction = str(next_subtask.get("instruction", "")).strip()
        max_steps = self._sanitize_max_steps(next_subtask.get("max_steps"))
        return {
            "done": False,
            "next_subtask": {
                "instruction": instruction,
                "max_steps": max_steps,
            },
        }

    def _sanitize_max_steps(self, value: Any) -> int:
        try:
            steps = int(value)
        except (TypeError, ValueError):
            steps = self.default_subtask_steps

        steps = max(self.min_subtask_steps, steps)
        steps = min(self.max_subtask_steps, steps)
        return steps

    def _fallback_instruction_from_text(self, text: str) -> str:
        """Extract a short salvageable instruction when JSON parsing fails."""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        for line in cleaned.splitlines():
            line = line.strip()
            if line:
                line = re.sub(r"^[-*\d\.\)\s]+", "", line)
                return line[:200]
        return "continue with the task"

    def _extract_first_json_object(self, text: str) -> str | None:
        """Extract the first balanced JSON object from arbitrary text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _image_dict_to_pil_list(self, image_dict: dict[str, torch.Tensor]) -> list[Image.Image]:
        """
        Convert OpenTau image tensors to PIL images.

        Expected tensor shape per image: [1, C, H, W] with C in {1, 3}.
        """
        pil_images: list[Image.Image] = []
        for key in sorted(image_dict):
            pil_images.append(self._single_tensor_to_pil(image_dict[key]))

        if not pil_images:
            raise ValueError("image_dict is empty; at least one image is required for planning.")
        return pil_images

    def _single_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor shape [B, C, H, W], got {tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError(
                f"QwenHighLevelPlanner currently expects batch_size=1 image tensors, got {tensor.shape[0]}"
            )

        image = tensor[0].detach().to("cpu")
        if image.shape[0] not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(image.shape)}")

        image = image.permute(1, 2, 0)
        if image.dtype.is_floating_point:
            image = image.float()
            if float(image.min().item()) < 0.0:
                image = (image + 1.0) / 2.0
            image = image.clamp(0.0, 1.0)
            image = (image * 255.0).round().to(torch.uint8)
        else:
            image = image.to(torch.uint8)

        np_image = image.numpy()
        if np_image.shape[2] == 1:
            return Image.fromarray(np_image[:, :, 0])
        return Image.fromarray(np_image)
