Hierarchical Evaluation with Qwen Planner
========================================

This guide describes how to run the minimal online hierarchical evaluation flow
added on top of OpenTau's existing LIBERO + VLA stack.

Overview
--------

The hierarchical evaluation path uses three layers:

1. a Qwen3-VL planner that emits exactly one next subtask,
2. a low-level VLA policy such as :math:`\pi_{0.5}`,
3. a LIBERO environment for rollout and success checking.

The planner only decides the next subtask. The low-level policy then executes
that subtask for a fixed number of environment steps. Once the step budget is
consumed, the planner is called again on the latest observation.

Relevant files
--------------

- ``src/opentau/agents/hierarchical_agent.py``
- ``src/opentau/planner/qwen3_vl_planner.py``
- ``src/opentau/planner/qwen_prompts.yaml``
- ``src/opentau/scripts/hierarchical_eval.py``
- ``configs/examples/pi05_hierarchical_eval_config.json``

Saved outputs
-------------

The hierarchical evaluator saves:

- one JSON summary per episode under ``hierarchical-eval/.../episode_XXXX.json``,
- one ``overall.json`` file for aggregated metrics.

Both now include a ``hierarchical`` section that records the active planner
settings, including the selected Qwen model, prompt keys, step budgets, and
history budget.

How to run
----------

Use the dedicated example config:

.. code-block:: bash

   opentau-hierarchical-eval --config_path configs/examples/pi05_hierarchical_eval_config.json

Configuration knobs
-------------------

The hierarchical planner is configured through ``cfg.hierarchical``:

.. code-block:: json

   "hierarchical": {
       "model_name": "Qwen/Qwen3-VL-4B-Instruct",
       "subtask_steps": 15,
       "min_subtask_steps": 5,
       "max_subtask_steps": 30,
       "max_subtasks": 20,
       "max_history_items": 10,
       "prompt_library_path": "src/opentau/planner/qwen_prompts.yaml",
       "system_prompt_key": "qwen_manipulation_short_system",
       "user_prompt_key": "qwen_manipulation_short_user"
   }

Recommended tuning
------------------

``subtask_steps``
~~~~~~~~~~~~~~~~~

This is the default per-subtask execution budget. Lower values lead to more
frequent replanning. Higher values let the low-level policy commit to the
current subtask longer.

- Start with ``10`` to ``20`` for manipulation tasks.
- Decrease it if the agent tends to overshoot or gets stuck following stale
  subtasks.
- Increase it if replanning is too frequent and the policy never gets enough
  time to complete approach or transport phases.

``min_subtask_steps`` and ``max_subtask_steps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These clamp the planner's returned ``max_steps`` value.

- Use a lower ``min_subtask_steps`` when you want fast correction loops.
- Use a higher ``max_subtask_steps`` when subtasks like carrying or placing
  typically require longer execution windows.
- A good starting range for manipulation is ``5`` to ``30``.

``max_subtasks``
~~~~~~~~~~~~~~~~

This is a hard cap on the number of replanning rounds per episode.

- Increase it for long-horizon tasks.
- Decrease it if you want to stop early when planning loops become unproductive.

``max_history_items``
~~~~~~~~~~~~~~~~~~~~~

This controls how many completed subtasks are shown to the planner.

- Use ``4`` to ``8`` when you want shorter prompts and faster planning.
- Use ``8`` to ``12`` when later subtasks depend on more execution history.
- If the model starts repeating stale instructions, reducing this value can
  help.

Prompt template selection
-------------------------

The planner prompt is loaded from ``src/opentau/planner/qwen_prompts.yaml``.

Currently two prompt styles are available:

``qwen_online_planner_system`` / ``qwen_online_planner_user``
    A more general online next-subtask planner prompt.

``qwen_manipulation_short_system`` / ``qwen_manipulation_short_user``
    A manipulation-oriented prompt that biases the model toward shorter
    imperative instructions such as ``approach the block`` or
    ``align gripper to handle``.

``qwen_manipulation_conservative_system`` /
``qwen_manipulation_conservative_user``
    A more conservative manipulation prompt that prefers approach/alignment
    before issuing grasp, release, or place actions.

For manipulation tasks, the shorter prompt pair is the recommended default.
If the planner tends to overcommit to grasping or repeatedly alternates between
grasp/release-style commands, try the conservative prompt pair.

Prompt tuning suggestions
-------------------------

If planning outputs are too long:

- use the ``qwen_manipulation_short_*`` templates,
- reduce ``max_history_items``,
- reduce ``max_subtask_steps`` if the planner keeps issuing overly broad
  subtasks.

If planning outputs are too myopic:

- increase ``max_history_items``,
- increase ``subtask_steps`` slightly,
- or switch back to the more general ``qwen_online_planner_*`` templates.

If planning outputs are too aggressive:

- try ``qwen_manipulation_conservative_*``,
- reduce ``max_subtask_steps`` so the planner revisits the scene earlier,
- or reduce ``max_history_items`` if the model keeps anchoring on stale grasp
  attempts.

Practical starting points
-------------------------

For short manipulation tasks in LIBERO:

- ``subtask_steps = 12`` to ``18``
- ``min_subtask_steps = 5``
- ``max_subtask_steps = 25`` to ``35``
- ``max_subtasks = 15`` to ``25``
- ``max_history_items = 6`` to ``10``
- prompt pair: ``qwen_manipulation_short_system`` /
  ``qwen_manipulation_short_user``

If the model grasps too early or repeats high-risk end-effector actions:

- keep ``subtask_steps`` in the lower half of the range,
- keep ``max_subtask_steps`` moderate,
- switch to ``qwen_manipulation_conservative_system`` /
  ``qwen_manipulation_conservative_user``.

For noisier or longer tasks:

- increase ``max_subtasks``,
- increase ``max_history_items``,
- keep ``subtask_steps`` moderate rather than very large.

Current limitations
-------------------

The minimal hierarchical implementation is intentionally simple:

- it currently supports ``eval.batch_size = 1`` only,
- it does not use a value model,
- it replans only when the current subtask budget is exhausted,
- it does not yet include explicit failure recovery heuristics.
