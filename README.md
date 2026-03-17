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

## 与上游 OpenTau 官方代码的主要差异

本仓库在 [OpenTau 官方仓库](https://github.com/TensorAuto/OpenTau) 基础上进行了以下扩展：

### 1. 新增：基于 Qwen3-VL 的层级任务规划评估流程

新增了一套完整的在线层级评估管线，使用视觉语言模型（Qwen3-VL）作为高层规划器，将长时域任务分解为可执行的子任务，再交由低层 VLA 策略（如 π₀.₅）执行。

| 组件 | 文件 | 说明 |
|---|---|---|
| 层级智能体 | `src/opentau/agents/hierarchical_agent.py` | 滚动式单子任务规划智能体，支持恢复性重规划 |
| Qwen3-VL 规划器 | `src/opentau/planner/qwen3_vl_planner.py` | 基于 Qwen3-VL-4B-Instruct 的高层规划器，利用视觉理解进行场景分析 |
| 提示词模板 | `src/opentau/planner/qwen_prompts.yaml` | 外置的 YAML 提示词库，包含 3 种风格的提示词 |
| 评估脚本 | `src/opentau/scripts/hierarchical_eval.py` | 层级评估入口脚本，支持在 LIBERO 环境中运行 |
| 配置文件 | `configs/examples/pi05_hierarchical_eval_config.json` | LIBERO 层级评估的示例配置 |
| 配置数据类 | `src/opentau/configs/default.py` | `HierarchicalConfig` 数据类，包含可调节的规划参数 |

**核心特性：**
- 三种提示词风格：通用型（general）、简短操作型（manipulation-short）、保守操作型（manipulation-conservative）
- 可配置的子任务步数预算、重规划次数上限、历史记录窗口大小
- 主规划失败时自动触发恢复性重规划（recovery replanning）
- 每个 episode 生成 JSON 摘要文件，包含完整的智能体状态信息，便于调试

详细使用说明和参数调优建议见 [层级评估指南](docs/source/tutorials/hierarchical_evaluation.rst)。

### 2. 修改：离线本地模型加载

所有 HuggingFace 远程模型引用已替换为本地路径（`/home/yjc/models/`），支持在无外网环境下部署和运行：

| 模型 | 原始 HuggingFace ID | 本地路径 |
|---|---|---|
| PaliGemma VLM 骨干网络 | `google/paligemma-3b-pt-224` | `/home/yjc/models/paligemma-3b-pt-224` |
| Qwen3-VL 高层规划器 | `Qwen/Qwen3-VL-4B-Instruct` | `/home/yjc/models/Qwen3-VL-4B-Instruct` |
| Fast 离散动作 tokenizer | `physical-intelligence/fast` | `/home/yjc/models/fast` |

此外，`src/opentau/utils/hub.py` 中新增了 `get_paligemma_source()` 函数，支持通过环境变量 `OPENTAU_PALIGEMMA_ID` 动态覆盖 PaliGemma 模型路径。

### 3. 优化：规划器提示词改进

`qwen_prompts.yaml` 中所有三种提示词风格均进行了优化：
- **更严格的完成判断**：规划器必须通过图像视觉确认任务的所有部分都已完成，才能返回 `{"done": true}`
- **多物体感知**：当任务提及"both"或"all"时，明确要求逐一检查每个目标物体的状态
- **失败重试逻辑**：如果之前的子任务可能未成功执行，规划器会重试而不是跳过
- **视觉验证提醒**：用户提示词中增加了"不要因为子任务已经尝试过就假定任务完成"的明确要求

## Acknowledgements

This project builds on the $\pi$ series of [papers][3] and many other open-source efforts—especially [LeRobot][4]—for re-implementing the $\pi$ models and helping standardize training infrastructure. OpenTau extends these foundations to provide a more accessible, comprehensive toolchain for training vision-language-action agents.

[1]:	https://www.tensor.ai
[2]:	https://huggingface.co/TensorAuto/tPi0.5-libero
[3]:	https://www.pi.website/blog
[4]:	https://huggingface.co/lerobot
[5]:    https://huggingface.co/TensorAuto/pi05_base
