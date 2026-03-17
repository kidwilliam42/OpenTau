<p align="center">
  <a href="https://www.tensor.auto">
    <img src="assets/logo.png" alt="Logo">
  </a>
</p>

<p align="center">
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/cpu_test.yml?query=branch%3Amain"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/cpu_test.yml/badge.svg?branch=main" alt="CPU Tests"></a>
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/gpu_test.yml"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/gpu_test.yml/badge.svg" alt="Nightly GPU Tests"></a>
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/regression_test.yml"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/regression_test.yml/badge.svg" alt="Nightly Regression Tests"></a>
  <a href="https://opentau.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/opentau/badge/?version=latest" alt="Documentation"></a>
  <a href="https://pypi.org/project/opentau/"><img src="https://img.shields.io/pypi/v/opentau" alt="Version"></a>
  <a href="https://pypi.org/project/opentau/"><img src="https://img.shields.io/pypi/status/opentau" alt="Status"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/opentau" alt="Python versions"></a>
  <a href="https://github.com/TensorAuto/OpenTau/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://hub.docker.com/r/tensorauto/opentau"><img src="https://img.shields.io/docker/v/tensorauto/opentau?label=Docker" alt="Docker"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"></a>
</p>

# OpenTau - Train VLA models with state-of-the-art techniques by Tensor

At Tensor, we are pushing the frontier of large foundation models for physical AI. In robot learning, a vision-language-action (VLA) model is a multimodal foundation model that integrates vision, language, and action. Today, VLA represents the leading approach for embodied AI, spanning autonomous driving, robot manipulation, and navigation.

OpenTau is Tensor’s open-source training toolchain for frontier VLA models—designed to make training reproducible, accessible, and scalable. At Tensor, we believe in open research and reproducible progress for the robotics community. By open-sourcing our training toolchain, we aim to expand knowledge sharing and accelerate scientific progress that others can reproduce.

Whether you use the official OpenPi codebase or LeRobot’s reimplementation, you may still be missing key components. OpenTau implements these key capabilities in one place:

- Co-training on an adjustable mixture of heterogeneous datasets
- Discrete actions for fast VLM convergence in $\pi_{0.5}$
- Knowledge insulation between the VLM backbone and the action expert
- Dropout in the VLM to reduce overfitting
- A reinforcement learning pipeline described in $\pi^*_{0.6}$
- And more...

OpenTau ($\tau$) is a tool developed by *[Tensor][1]* to bridge this gap, and we also use it internally to train our proprietary in-house models. Our goal is to help you train VLAs on any dataset while fully leveraging state-of-the-art techniques. We plan to continuously upgrade this repository to keep pace with the state of the art in the robotics community.

|                                                 Features |         OpenPi          |             LeRobot              | **OpenTau** |
|---------------------------------------------------------:|:-----------------------:|:--------------------------------:|:-----------:|
|                  Co-training with Heterogeneous Datasets |            ❌            |                ❌                 |      ✅      |
|                 Discrete Actions Training in $\pi_{0.5}$ |            ❌            |                ❌                 |      ✅      |
| Knowledge Insulation (KI) between VLM and Action Decoder |            ❌            |                ❌                 |      ✅      |
|                              Dropout Layers in PaliGemma | ✅ (Jax) <br>❌ (PyTorch)|                ❌                |      ✅     |
|                        Multi-Node and Multi-GPU Training |            ❌            |                ✅                 |      ✅      |
|                 Fully Functioning $\pi_{0.5}$ Checkpoint |            ✅            | ❌ <br> (Missing Text Embeddings) |      ✅      |
|                       Visualize dataset with URDF models |            ❌            |                ❌                 |      ✅      |
|            Simulation Environments for Evaluating Models |            ❌            |                ✅                 |      ✅      |
|                 Create Validation Splits During Training |            ❌            |                ❌                 |      ✅      |
|    $\pi^{*}_{0.6}$ style Reinforcement Learning Pipeline |            ❌            |                ❌                 |      ✅      |
|                        Post-training on Human Data (Beta)|            ❌            |                ❌                 |      ✅      |
|                                                Framework |      Jax / PyTorch       |             PyTorch               |   PyTorch    |

## Quick Start
If you are familiar with LeRobot, getting started with OpenTau is very easy.
Because OpenTau is a fork of the popular LeRobot repository, any LeRobot-compliant policy and dataset can be used directly with OpenTau.
Check out our [documentation](https://opentau.readthedocs.io/) to get started quickly.
We provide a [quick start guide](https://opentau.readthedocs.io/en/latest/getting_started.html) to help you get started with OpenTau.

For using local notebooks to train and evaluate models, find the notebooks at [notebooks/pi05_training.ipynb](https://github.com/TensorAuto/OpenTau/blob/main/notebooks/pi05_training.ipynb) and [notebooks/pi05_evaluation_only.ipynb](https://github.com/TensorAuto/OpenTau/blob/main/notebooks/pi05_evaluation_only.ipynb).

For using the Google Colab notebooks to train and evaluate models, find the colab notebooks here: [pi05_training](https://colab.research.google.com/drive/1DeU0lNnEzs1KHo0Nkgh4YKBr-xu9moBM?usp=sharing) and [pi05_evaluation_only](https://colab.research.google.com/drive/1U_AyuH9WYMT4anEWvsOtIT7g01jA0WGm?usp=sharing) respectively.

## Checkpoints
We provide fully functioning $\pi_{0.5}$ checkpoints trained with high success rates. We plan to release more models in the near future.

| Model Checkpoint              | Description                                                                                                   | Success Rate (%)                                                   |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [TensorAuto/tPi0.5-libero][2] | A $\pi_{0.5}$ model checkpoint trained on the LIBERO dataset with discrete actions and knowledge insulation.  | 98.4% (10) <br> 97.6% (Goal) <br> 100% (Object) <br> 98% (Spatial) |
| [TensorAuto/pi05_base][5]     | A $\pi_{0.5}$ model checkpoint converted from the official openpi checkpoint, with language embeddings added. | N/A                                                                |
| More coming soon...           |                                                                                                               |                                                                    |

## Differences from Upstream OpenTau

This fork extends the [official OpenTau repository](https://github.com/TensorAuto/OpenTau) with the following changes:

### 1. Hierarchical Evaluation with Qwen3-VL Planner

Added a complete online hierarchical evaluation pipeline that decomposes long-horizon tasks into subtasks using a vision-language model as a high-level planner.

| Component | File | Description |
|---|---|---|
| Hierarchical Agent | `src/opentau/agents/hierarchical_agent.py` | Rolling single-subtask planning agent with recovery replanning |
| Qwen3-VL Planner | `src/opentau/planner/qwen3_vl_planner.py` | High-level planner using Qwen3-VL-4B-Instruct for scene understanding |
| Prompt Templates | `src/opentau/planner/qwen_prompts.yaml` | Externalized YAML prompt library with 3 prompt styles |
| Eval Script | `src/opentau/scripts/hierarchical_eval.py` | Entry point for hierarchical evaluation on LIBERO |
| Config | `configs/examples/pi05_hierarchical_eval_config.json` | Example configuration for LIBERO hierarchical eval |
| Config Dataclass | `src/opentau/configs/default.py` | `HierarchicalConfig` with tunable planning parameters |

**Key features:**
- Three prompt styles: general, manipulation-short, and manipulation-conservative
- Configurable subtask step budgets, replanning limits, and history window
- Recovery replanning when primary planning returns empty results
- Per-episode JSON summaries with full agent state for debugging

See [Hierarchical Evaluation Guide](docs/source/tutorials/hierarchical_evaluation.rst) for detailed usage and tuning recommendations.

### 2. Offline / Local Model Loading

All HuggingFace model references have been replaced with local paths under `/home/yjc/models/` to support air-gapped (no internet) deployment:

| Model | Original ID | Local Path |
|---|---|---|
| PaliGemma VLM backbone | `google/paligemma-3b-pt-224` | `/home/yjc/models/paligemma-3b-pt-224` |
| Qwen3-VL planner | `Qwen/Qwen3-VL-4B-Instruct` | `/home/yjc/models/Qwen3-VL-4B-Instruct` |
| Fast tokenizer | `physical-intelligence/fast` | `/home/yjc/models/fast` |

Additionally, `src/opentau/utils/hub.py` provides `get_paligemma_source()` which supports overriding the PaliGemma path via the `OPENTAU_PALIGEMMA_ID` environment variable.

### 3. Improved Planner Prompts

All three prompt styles in `qwen_prompts.yaml` have been updated with:
- **Stricter done-check**: planner must visually confirm ALL parts of the task are completed before returning `{"done": true}`
- **Multi-object awareness**: explicit rule to verify EACH object when the task mentions "both" or "all"
- **Retry logic**: if a previous subtask may not have succeeded, retry instead of skipping
- **Visual verification reminder**: user prompts now instruct the planner not to assume completion just because subtasks were attempted

## Acknowledgements

This project builds on the $\pi$ series of [papers][3] and many other open-source efforts—especially [LeRobot][4]—for re-implementing the $\pi$ models and helping standardize training infrastructure. OpenTau extends these foundations to provide a more accessible, comprehensive toolchain for training vision-language-action agents.

[1]:	https://www.tensor.ai
[2]:	https://huggingface.co/TensorAuto/tPi0.5-libero
[3]:	https://www.pi.website/blog
[4]:	https://huggingface.co/lerobot
[5]:    https://huggingface.co/TensorAuto/pi05_base
