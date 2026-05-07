import pytest
import torch

from opentau.planner.vlm_action_selector import (
    CANDIDATES,
    DEFAULT_QWEN3_VL_MODEL,
    Qwen3VLActionSelector,
    VLMActionSelector,
    build_select_action_prompt,
    normalize_current_label,
    token_id_from_tokenizer,
    vlm_select_action_by_logits,
)


class FakeVLM:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def forward(self, image, prompt):
        self.calls.append({"image": image, "prompt": prompt})
        return {"logits": self.logits}


class FakeTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def encode(self, label, add_special_tokens=False):
        assert not add_special_tokens
        return self.token_ids[label]


class FakeQwenProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer({label: [idx] for idx, label in enumerate(CANDIDATES)})
        self.messages = None
        self.processor_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        assert not tokenize
        assert add_generation_prompt
        return "chat-template-text"

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeQwenModel:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1

    def parameters(self):
        return iter(())

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"logits": self.logits}


def test_build_select_action_prompt_contains_fixed_candidates_and_task_context():
    prompt = build_select_action_prompt(
        current_label="B",
        task="Place the red cube into the target slot.",
        slot="S2",
    )

    assert "A = PICK" in prompt
    assert "B = NAV_SLOT" in prompt
    assert "C = PLACE" in prompt
    assert "D = NAV_BIN" in prompt
    assert "E = DONE" in prompt
    assert "Current action: B" in prompt
    assert "Task: Place the red cube into the target slot." in prompt
    assert "Target slot: S2" in prompt
    assert "Choose exactly one of A, B, C, D, E." in prompt


def test_normalize_current_label_converts_none_to_none_text_and_rejects_done():
    assert normalize_current_label(None) == "NONE"
    assert normalize_current_label("A") == "A"

    with pytest.raises(ValueError):
        normalize_current_label("E")


def test_vlm_select_action_by_logits_uses_fixed_candidate_argmax():
    token_ids = {label: idx for idx, label in enumerate(CANDIDATES)}
    logits = torch.tensor([[0.1, 1.0, 0.3, 0.4, 0.2]])
    vlm = FakeVLM(logits=logits)

    selected = vlm_select_action_by_logits(
        image="frame",
        current_label=None,
        task="Put the object in slot S2.",
        vlm=vlm,
        token_id_fn=token_ids.__getitem__,
        slot="S2",
    )

    assert selected == "B"
    assert vlm.calls[0]["image"] == "frame"
    assert "Current action: NONE" in vlm.calls[0]["prompt"]


def test_vlm_select_action_by_logits_uses_last_token_for_sequence_logits():
    token_ids = {label: idx for idx, label in enumerate(CANDIDATES)}
    logits = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.2, 0.9],
            ]
        ]
    )
    vlm = FakeVLM(logits=logits)

    selected = vlm_select_action_by_logits(
        image="frame",
        current_label="D",
        task="Put the object in slot S2.",
        vlm=vlm,
        token_id_fn=token_ids.__getitem__,
        slot="S2",
    )

    assert selected == "E"


def test_vlm_action_selector_wrapper_delegates_to_logits_selector():
    token_ids = {label: idx for idx, label in enumerate(CANDIDATES)}
    selector = VLMActionSelector(vlm=FakeVLM(torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0])), token_id_fn=token_ids.__getitem__)

    assert selector.select(image="frame", current_label="B", task="Place object.", slot="S2") == "C"


def test_token_id_from_tokenizer_requires_single_token_labels():
    tokenizer = FakeTokenizer({"A": [42], "B": [1, 2]})

    assert token_id_from_tokenizer(tokenizer, "A") == 42

    with pytest.raises(ValueError):
        token_id_from_tokenizer(tokenizer, "B")


def test_qwen3_vl_action_selector_scores_qwen_forward_logits():
    processor = FakeQwenProcessor()
    model = FakeQwenModel(torch.tensor([[[0.0, 0.1, 0.2, 2.5, 0.3]]]))
    selector = Qwen3VLActionSelector(
        model_name=DEFAULT_QWEN3_VL_MODEL,
        device="cpu",
        processor=processor,
        model=model,
    )

    selected = selector.select(
        image="pil-image",
        current_label="B",
        task="Put the object into slot S2.",
        slot="S2",
    )

    assert selected == "D"
    assert model.eval_calls == 1
    assert model.calls[0]["input_ids"].shape == (1, 3)
    assert processor.processor_kwargs["text"] == ["chat-template-text"]
    assert processor.processor_kwargs["images"] == ["pil-image"]
    user_content = processor.messages[0]["content"]
    assert user_content[0] == {"type": "image", "image": "pil-image"}
    assert user_content[1]["type"] == "text"
    assert "Current action: B" in user_content[1]["text"]
    assert "Target slot: S2" in user_content[1]["text"]
