from .label_action_loop import (
    Camera,
    LabelActionLoop,
    LabelActionSelector,
    LabelDecision,
    LabelLoopResult,
    LabelLoopStep,
    Pi05Executor,
    decide_label_transition,
    to_pi05_instruction,
)
from .ros_label_action_loop import (
    RosImageTopicCamera,
    RosInstructionExecutor,
    RosLabelActionLoopConfig,
    compressed_image_msg_to_pil,
    create_ros_label_action_loop,
    image_msg_to_pil,
    run_ros_label_action_loop,
)

__all__ = [
    "Camera",
    "ExecutionRecord",
    "HierarchicalAgent",
    "LabelActionLoop",
    "LabelActionSelector",
    "LabelDecision",
    "LabelLoopResult",
    "LabelLoopStep",
    "Pi05Executor",
    "RosImageTopicCamera",
    "RosInstructionExecutor",
    "RosLabelActionLoopConfig",
    "SubtaskPlan",
    "compressed_image_msg_to_pil",
    "create_ros_label_action_loop",
    "image_msg_to_pil",
    "run_ros_label_action_loop",
    "decide_label_transition",
    "to_pi05_instruction",
]


def __getattr__(name: str):
    """Lazily load torch-dependent hierarchical agent classes."""
    if name in {"ExecutionRecord", "HierarchicalAgent", "SubtaskPlan"}:
        from .hierarchical_agent import ExecutionRecord, HierarchicalAgent, SubtaskPlan

        return {
            "ExecutionRecord": ExecutionRecord,
            "HierarchicalAgent": HierarchicalAgent,
            "SubtaskPlan": SubtaskPlan,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
