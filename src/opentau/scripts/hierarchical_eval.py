#!/usr/bin/env python

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from opentau.agents.hierarchical_agent import HierarchicalAgent
from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.envs.factory import make_envs
from opentau.envs.utils import close_envs, preprocess_observation
from opentau.planner.qwen3_vl_planner import QwenHighLevelPlanner
from opentau.policies.factory import make_policy
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device, init_logging

def _get_single_env(envs: dict[str, dict[int, Any]]):
    """
    Minimal hierarchical eval only supports one vectorized env entry.

    make_envs(...) returns:
        dict[suite_name][task_id] -> gym.vector.VectorEnv
    """
    if len(envs) != 1:
        raise NotImplementedError(
            f"hierarchical_eval currently expects exactly one env suite, got {list(envs.keys())}"
        )

    suite_name = next(iter(envs.keys()))
    suite_envs = envs[suite_name]
    if len(suite_envs) != 1:
        raise NotImplementedError(
            f"hierarchical_eval currently expects exactly one task_id, got {list(suite_envs.keys())}"
        )

    task_id = next(iter(suite_envs.keys()))
    return suite_name, task_id, suite_envs[task_id]


def _get_env_task_text(env) -> str:
    """Fetch the environment language description exposed by the wrapper."""
    task_desc = env.call("get_wrapper_attr", "task_description")
    if len(task_desc) > 0 and isinstance(task_desc[0], str) and task_desc[0]:
        return task_desc[0]

    task_name = env.call("get_wrapper_attr", "task")
    if len(task_name) > 0 and isinstance(task_name[0], str) and task_name[0]:
        return task_name[0]

    raise ValueError("Could not infer task text from environment.")


def _save_episode_summary(
    output_dir: Path,
    episode_idx: int,
    suite_name: str,
    task_id: int,
    task_text: str,
    episode_success: bool,
    agent_state: dict[str, Any],
    episode_reward_sum: float,
    episode_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_idx": episode_idx,
        "suite_name": suite_name,
        "task_id": task_id,
        "task_text": task_text,
        "success": episode_success,
        "episode_reward_sum": episode_reward_sum,
        "episode_steps": episode_steps,
        "agent_state": agent_state,
    }

    with open(output_dir / f"episode_{episode_idx:04d}.json", "w") as f:
        json.dump(payload, f, indent=2)


def _run_single_episode(
    env,
    cfg: TrainPipelineConfig,
    planner: QwenHighLevelPlanner,
    low_level_policy,
    episode_idx: int,
    task_text: str,
    planner_default_subtask_steps: int,
    max_subtasks: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run one episode with online single-subtask planning.

    Assumptions:
      - batch_size = 1
      - planner emits one subtask at a time
      - replanning happens after the current subtask budget is exhausted
    """
    observation, _ = env.reset(seed=[seed] if seed is not None else None)
    agent = HierarchicalAgent(
        planner=planner,
        low_level_policy=low_level_policy,
        task=task_text,
        default_subtask_steps=planner_default_subtask_steps,
        max_subtasks=max_subtasks,
    )
    agent.reset()

    done = False
    step = 0
    success = False
    episode_reward_sum = 0.0
    max_steps = env.call("_max_episode_steps")[0]

    logging.info(f"[episode {episode_idx}] task: {task_text}")

    with torch.inference_mode():
        while not done and not agent.is_finished() and step < max_steps:
            observation_t = preprocess_observation(observation, cfg=cfg)

            if agent.needs_planning:
                logging.info(f"[episode {episode_idx}] replanning from latest observation")

            action = agent.act(observation_t)
            action_numpy = action.to("cpu").numpy()
            if action_numpy.ndim != 2:
                raise ValueError(f"Expected action shape [B, action_dim], got {action_numpy.shape}")

            observation, reward, terminated, truncated, info = env.step(action_numpy)
            agent.update_after_env_step(info)

            reward_value = float(np.asarray(reward).reshape(-1)[0])
            episode_reward_sum += reward_value
            step += 1

            if "is_success" in info:
                success = bool(np.asarray(info["is_success"]).reshape(-1)[0])
            done = bool(np.any(terminated)) or bool(np.any(truncated)) or success

            logging.info(
                "[episode %s] step=%s reward=%.4f success=%s current_subtask=%s",
                episode_idx,
                step,
                reward_value,
                success,
                agent.current_instruction(),
            )

    return {
        "success": success,
        "episode_reward_sum": episode_reward_sum,
        "episode_steps": step,
        "agent_state": agent.debug_state(),
    }


@parser.wrap()
def hierarchical_eval_main(cfg: TrainPipelineConfig):
    """
    Minimal entry point for online hierarchical evaluation.

    Constraints:
      - cfg.env must be set
      - cfg.policy must be set
      - cfg.eval.batch_size must be 1
      - current implementation supports one suite and one task_id only
    """
    init_logging(level=logging.DEBUG if cfg.debug else logging.INFO)
    logging.info(pformat(asdict(cfg)))

    if cfg.env is None:
        raise ValueError("cfg.env must be set for hierarchical_eval.")
    if cfg.policy is None:
        raise ValueError("cfg.policy must be set for hierarchical_eval.")
    if cfg.eval is None:
        raise ValueError("cfg.eval must be set for hierarchical_eval.")
    if cfg.eval.batch_size != 1:
        raise NotImplementedError(
            f"hierarchical_eval supports cfg.eval.batch_size=1 only, got {cfg.eval.batch_size}"
        )

    device = auto_torch_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Making environment.")
    envs = make_envs(
        cfg.env,
        cfg,
        n_envs=cfg.eval.batch_size,
        use_async_envs=False,
    )
    suite_name, task_id, env = _get_single_env(envs)
    task_text = _get_env_task_text(env)
    logging.info(f"Resolved env task text: {task_text}")

    logging.info("Making low-level policy.")
    low_level_policy = make_policy(cfg=cfg.policy)
    low_level_policy.to(device=device, dtype=torch.bfloat16)
    low_level_policy.eval()

    logging.info("Making Qwen high-level planner.")
    planner = QwenHighLevelPlanner(
        model_name=cfg.hierarchical.model_name,
        device=device,
        default_subtask_steps=cfg.hierarchical.subtask_steps,
        min_subtask_steps=cfg.hierarchical.min_subtask_steps,
        max_subtask_steps=cfg.hierarchical.max_subtask_steps,
        max_history_items=cfg.hierarchical.max_history_items,
        prompt_library_path=cfg.hierarchical.prompt_library_path,
        system_prompt_key=cfg.hierarchical.system_prompt_key,
        user_prompt_key=cfg.hierarchical.user_prompt_key,
    )

    base_output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else Path("outputs")
    details = f"{cfg.env.type}-{cfg.env.task}-hierarchical-{cfg.eval.n_episodes}"
    output_dir = base_output_dir / "hierarchical-eval" / details
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_results = []
    try:
        for episode_idx in range(cfg.eval.n_episodes):
            episode_seed = None if cfg.seed is None else cfg.seed + episode_idx
            result = _run_single_episode(
                env=env,
                cfg=cfg,
                planner=planner,
                low_level_policy=low_level_policy,
                episode_idx=episode_idx,
                task_text=task_text,
                planner_default_subtask_steps=cfg.hierarchical.subtask_steps,
                max_subtasks=cfg.hierarchical.max_subtasks,
                seed=episode_seed,
            )
            episode_results.append(result)

            _save_episode_summary(
                output_dir=output_dir,
                episode_idx=episode_idx,
                suite_name=suite_name,
                task_id=task_id,
                task_text=task_text,
                episode_success=result["success"],
                agent_state=result["agent_state"],
                episode_reward_sum=result["episode_reward_sum"],
                episode_steps=result["episode_steps"],
            )

        overall = {
            "success_rate": float(np.mean([r["success"] for r in episode_results]))
            if episode_results
            else 0.0,
            "avg_reward": float(np.mean([r["episode_reward_sum"] for r in episode_results]))
            if episode_results
            else 0.0,
            "avg_steps": float(np.mean([r["episode_steps"] for r in episode_results]))
            if episode_results
            else 0.0,
            "n_episodes": len(episode_results),
        }

        with open(output_dir / "overall.json", "w") as f:
            json.dump({"overall": overall, "episodes": episode_results}, f, indent=2)

        logging.info(f"Hierarchical eval overall: {overall}")
    finally:
        close_envs(envs)

    logging.info("End of hierarchical eval")


if __name__ == "__main__":
    hierarchical_eval_main()
